"""
Count targets per anchor text in Wikipedia paragraphs
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys, pathlib, argparse, logging, json, collections, re, html

import dawg
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

from .normalize import normalize


def count_links(lines, language=None):
    link_count = collections.defaultdict(collections.Counter)
    for line in lines:
        pagetitle, links, paragraph = line.split("\t", 2)
        links = json.loads(links)
        for a, e in links.items():
            # Clean the anchor text before counting
            for n in normalize(a, language=language):
                link_count[n][e] += 1
    return [(a, e, c) for a, cs in link_count.items() for e, c in cs.items()]


def count(paragraphlinks: pathlib.Path, min_count: int = 2, language: str = None):
    """
    Count targets per anchor text in Wikipedia paragraphs.

    Writes `count.min{min_count}.json`

    Args:
        paragraphlinks: Directory of (pagetitle, links-json, paragraph) .tsv files

    Keyword Arguments:
        min_count: Minimal (anchor-text, target) occurrence
        language: Language code for tokenization & stemming
    """

    import dask.bag as db
    from .scale import progress, get_client
    
    if language:
        logging.info(f"Snowball stemming for language: {language}")

    with get_client():

        bag = db.read_text(str(paragraphlinks) + "/*", files_per_partition=3)
        counts = (
            bag.map_partitions(count_links, language=language)
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
        
        stem = f'.stem-{language}' if language else ''
        outfile = paragraphlinks.parent / f"count.min{min_count}{stem}.json"
        logging.info(f"Writing to {outfile}")
        a_e_count.compute().to_json(outfile)
