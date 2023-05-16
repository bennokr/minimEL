"""
Count targets per anchor text in Wikipedia paragraphs
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys, pathlib, argparse, logging, json, collections, re, html, itertools

try:
    import dawg
except ImportError:
    import dawg_python as dawg
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

from .normalize import normalize
from minimel.vectorize import vw_tok

def count_links(lines, stem=None, head=None):
    link_count = collections.defaultdict(collections.Counter)
    for line in itertools.islice(lines, 0, head):
        pagetitle, links, paragraph = line.split("\t", 2)
        links = json.loads(links)
        for a, e in links.items():
            # Clean the anchor text before counting
            for n in normalize(a, language=stem):
                link_count[n][e] += 1
    return [(a, e, c) for a, cs in link_count.items() for e, c in cs.items()]


def count(paragraphlinks: pathlib.Path, *, min_count: int = 2, stem: str = None, head: int = None):
    """
    Count targets per anchor text in Wikipedia paragraphs.

    Writes `count.min{min_count}[.stem-{LANG}].json`

    Args:
        paragraphlinks: Directory of (pagetitle, links-json, paragraph) .tsv files

    Keyword Arguments:
        min_count: Minimal (anchor-text, target) occurrence
        stem: Language code for tokenization & stemming
        head: Use only N first lines from each partition
    """

    import dask.bag as db
    from .scale import progress, get_client
    
    if stem:
        logging.info(f"Snowball stemming for language: {stem}")

    with get_client():

        bag = db.read_text(str(paragraphlinks) + "/*", files_per_partition=3)
        counts = (
            bag.map_partitions(count_links, stem=stem, head=head)
            .to_dataframe(meta={'a':str, 'e':int, 'c':int})
            .groupby(["a", "e"])["c"].sum(split_out=32)
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

        a_e_count = select.groupby("a").apply(make_id_count_dict, meta=('c', int))
        if logging.root.level < 30:
            progress(a_e_count.persist(), out=sys.stderr)
        
        s = f'-stem' if stem else ''
        outfile = paragraphlinks.parent / f"count.min{min_count}{s}.json"
        logging.info(f"Writing to {outfile}")
        a_e_count.compute().to_json(outfile)

        

def get_matches(surface_trie, text, stem=None):
    # TODO: is normalize best here?
    for normtext in normalize(text, language=stem):
        normtoks = normtext.split()
        for i,tok in enumerate(normtoks):
            for comp in surface_trie.keys(tok):
                comp_toks = comp.split()
                if normtoks[i:i+len(comp_toks)] == comp_toks:
                    yield comp

def count_surface_lines(lines, countfile, stem=None, head=None):
    surfaces = json.load(open(countfile))
    surface_trie = dawg.CompletionDAWG(surfaces)
    counts = collections.Counter()
    for line in itertools.islice(lines, 0, head):
        _, _, text = line.split('\t', 2)
        counts.update(get_matches(surface_trie, text, stem=stem))
    return list(counts.items())
        

def count_surface(paragraphlinks: pathlib.Path, countfile: pathlib.Path, *, stem: str = None, head: int = None):
    """
    Count anchor texts in Wikipedia paragraphs.

    Writes `word{countfile}[.stem-{LANG}].json`

    Args:
        paragraphlinks: Directory of (pagetitle, links-json, paragraph) .tsv files
        countfile: Hyperlink anchor count JSON file

    Keyword Arguments:
        stem: Language code for tokenization & stemming
        head: Use only N first lines from each partition
    """

    import dask.bag as db
    from .scale import progress, get_client
    
    if stem:
        logging.info(f"Snowball stemming for language: {stem}")

    with get_client():

        bag = db.read_text(str(paragraphlinks) + "/*", files_per_partition=3)
        counts = (
            bag.map_partitions(count_surface_lines, countfile, stem=stem, head=head)
            .to_dataframe(meta={'surface':str, 'c':int})
            .groupby("surface")["c"].sum(split_out=32)
            .persist()
        )

        logging.info("Counting surfaceforms...")
        if logging.root.level < 30:
            progress(counts.persist(), out=sys.stderr)

        logging.info(f"Got {len(counts)} counts.")
        logging.info("Aggregating...")

        if logging.root.level < 30:
            progress(counts.persist(), out=sys.stderr)
        
        stem = f'-stem' if stem else ''
        outfile = paragraphlinks.parent / f"word{countfile.stem}{stem}.json"
        logging.info(f"Writing to {outfile}")
        counts.compute().to_json(outfile)
