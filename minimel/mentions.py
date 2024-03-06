"""
Find entity mentions
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys, pathlib, json, collections, itertools, logging

from .normalize import normalize

log = logging.getLogger(__name__)


def get_matches(matcher, text, offsets=True, stem=None):
    # TODO stem
    for i, start, end in matcher.find_matches_as_indexes(text):
        # Make sure match is surrounded by non-alphanumeric characters
        if start != 0 and text[start - 1].isalnum():
            continue
        if end != len(text) and text[end].isalnum():
            continue
        t = text[start:end]
        yield (start, t) if offsets else t


def setup_matcher(countfile, names=None):
    from ahocorasick_rs import AhoCorasick, MatchKind, Implementation

    if not names:
        log.info(f"Setting up AhoCorasick from {countfile}")
        names = list(json.load(open(countfile)))
    matcher = AhoCorasick(
        names,
        matchkind=MatchKind.LeftmostLongest,
        implementation=Implementation.NoncontiguousNFA,
    )
    return matcher


def count_name_lines(lines, countfile, stem=None, head=None):
    matcher = setup_matcher(countfile)
    counts = collections.Counter()
    for line in itertools.islice(lines, 0, head):
        _, _, text = line.split("\t", 2)
        counts.update(get_matches(matcher, text.lower(), offsets=False, stem=stem))
    return list(counts.items())


def count_names(
    paragraphlinks: pathlib.Path,
    countfile: pathlib.Path,
    *,
    outfile: pathlib.Path = None,
    stem: str = None,
    head: int = None,
):
    """
    Count anchor texts in Wikipedia paragraphs.

    Args:
        paragraphlinks: Directory of (pagetitle, links-json, paragraph) .tsv files
        countfile: Hyperlink anchor count JSON file

    Keyword Arguments:
        outfile: Output file or directory (default: `name{countfile}[.stem-{LANG}].json`)
        stem: Stemming language ISO 639-1 (2-letter) code
        head: Use only N first lines from each partition
    """

    import dask.bag as db
    from .scale import progress, get_client

    if stem:
        logging.info(f"Snowball stemming for language: {stem}")

    with get_client():
        bag = db.read_text(str(paragraphlinks) + "/*", files_per_partition=3)
        counts = (
            bag.map_partitions(count_name_lines, countfile, stem=stem, head=head)
            .to_dataframe(meta={"name": str, "c": int})
            .groupby("name")["c"]
            .sum(split_out=32)
            .persist()
        )

        logging.info("Counting names...")
        if logging.root.level < 30:
            progress(counts.persist(), out=sys.stderr)

        logging.info(f"Got {len(counts)} counts.")
        logging.info("Aggregating...")

        if logging.root.level < 30:
            progress(counts.persist(), out=sys.stderr)

        s = f"-stem" if stem else ""
        h = f"-head{head}" if head else ""
        fname = f"word{countfile.stem}{s}{h}.json"
        if not outfile:
            outfile = paragraphlinks.parent / fname
        if outfile.is_dir():
            outfile = outfile / fname
        outfile.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing to {outfile}")
        counts.compute().to_json(outfile)
