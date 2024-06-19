"""
Count targets per anchor text in Wikipedia paragraphs
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys, pathlib, argparse, logging, json, collections, re, html, itertools

from .normalize import normalize
from minimel.vectorize import vw_tok


def count_links(lines, stem=None, head=None, split=None, fold=None):
    link_count = collections.defaultdict(collections.Counter)
    for i, line in enumerate(itertools.islice(lines, 0, head)):
        if split and (i % split == fold):
            continue
        pagetitle, links, paragraph = line.split("\t", 2)
        links = json.loads(links)
        for a, e in links.items():
            # Clean the anchor text before counting
            for n in normalize(a, language=stem):
                link_count[n][e] += 1
    return [(a, e, c) for a, cs in link_count.items() for e, c in cs.items()]


def count(
    paragraphlinks: pathlib.Path,
    *,
    outfile: pathlib.Path = None,
    min_count: int = 2,
    stem: str = None,
    head: int = None,
    split: int = None,
    fold: int = None,
):
    """
    Count targets per anchor text in Wikipedia paragraphs.

    Writes `count.min{min_count}[.stem-{LANG}].json`

    Args:
        paragraphlinks: Directory of (pagetitle, links-json, paragraph) .tsv files

    Keyword Arguments:
        outfile: Output file or directory (default: `count.json`)
        stem: Stemming language ISO 639-1 (2-letter) code
        min_count: Minimal (anchor-text, target) occurrence
        head: Use only N first lines from each partition
        split: Split the data into several parts
        fold: Ignore this fold of the split data
    """
    assert (not split) or (split > fold)

    import dask.bag as db
    from .scale import progress, get_client

    if stem:
        logging.info(f"Stemming for language: {stem}")

    with get_client():
        bag = db.read_text(str(paragraphlinks) + "/*", files_per_partition=3)
        counts = (
            bag.map_partitions(
                count_links, stem=stem, head=head, split=split, fold=fold
            )
            .to_dataframe(meta={"a": str, "e": int, "c": int})
            .groupby(["a", "e"])["c"]
            .sum(split_out=32)
            .persist()
        )

        logging.info("Counting links...")
        if logging.root.level < 30:
            progress(counts.persist(), out=sys.stderr)

        logging.info(f"Got {len(counts)} counts.")
        logging.info("Aggregating...")

        select = counts[counts >= min_count].to_frame().reset_index()
        # select = select[select.groupby('a')['e'].transform('nunique') > 1]

        def make_id_count_dict(x):
            return {f"Q{e}": c for e, c in x.set_index("e")["c"].items()}

        a_e_count = select.groupby("a").apply(make_id_count_dict, meta=("c", int))
        if logging.root.level < 30:
            progress(a_e_count.persist(), out=sys.stderr)

        s = f"-stem" if stem else ""
        fname = f"count.min{min_count}{s}.json"
        if not outfile:
            outfile = paragraphlinks.parent / fname
        if outfile.is_dir():
            outfile = outfile / fname
        outfile.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Writing to {outfile}")
        a_e_count.compute().to_json(outfile)
