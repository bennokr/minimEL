import typing
import pathlib
import logging
import os

from .get_disambig import get_disambig
from .count import count
from .vectorize import vectorize
from .train import train

def experiment(
    root: pathlib.Path = pathlib.Path('.'),
    *,
    nparts: int = 100,
    head: int = None,
    stem: bool = None,
    min_count: typing.List[int] = (2,),
    
):
    """
    Run all steps to train and evaluate EL models over a parameter sweep.
    
    The root directory must contain the following files:
    
    - `wikidata*-disambig.txt`: See :obj:`~get_disambig.get_disambig`
    
    Args: 
        root: Root directory
    
    Keyword Arguments:
        nparts: Number of parts to chunk wikidump into
        head: Use only N first lines from each partition
        stem: Use stemming (leave out for both, use --no-stem for no stemming)
        
    """
    logging.info(f"Running experiments for {root}")
    def find(directory, glob):
        try:
            found = next(directory.glob(glob)).absolute()
            logging.info(f"Using {found}")
            return found
        except StopIteration:
            raise Exception(f'File {glob} not found in {directory}')
    
    dawgfile = find(root, 'index_*.dawg')
    wikidump = find(root, '*-pages-articles.xml')
    
    # Get disambiguations
    if not any(root.glob('disambig.json')):
        wikidata_disambigfile = find(root, 'wikidata*-disambig.txt')
        logging.info(f"Getting disambiguations...")
        get_disambig(wikidump, dawgfile, wikidata_disambigfile, nparts=nparts)
    disambigfile = find(root, 'disambig.json')
    
    # Get paragraph links
    if not any(root.glob('*-paragraph-links')):
        logging.info(f"Getting paragraph links...")
        get_paragraphs(wikidump, dawgfile)
    paragraphlinks = find(root, '*-paragraph-links')
    
    # Count
    curdir = root
    stem = (True, False) if (stem is None) else (stem,)
    for p_stem in stem: # parameter sweep
        for p_min_count in min_count: # parameter sweep
            newdir = curdir / f'count__{"" if p_stem else "no-"}stem__min-count={p_min_count}'
            if not any(newdir.glob('count*.json')):
                newdir.mkdir(parents=True, exist_ok=True)
                os.chdir(newdir)
                logging.info(f"Counting in {newdir}...")
                count(paragraphlinks, min_count = p_min_count, stem = p_stem, head=head)
                os.chdir(curdir)
    
    for countfile in curdir.glob('count*.json'):
        print(countfile)