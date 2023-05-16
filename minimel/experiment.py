import typing
import pathlib
import logging

from .vectorize import vectorize
from .train import train
from .get_disambig import get_disambig

def experiment(
    root: pathlib.Path = pathlib.Path('.'),
    *,
    nparts: int = 100,
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
            found = next(directory.glob(glob))
            logging.info(f"Using {found}")
            return found
        except StopIteration:
            raise Exception(f'File {glob} not found in {directory}')
    
    dawgfile = find(root, 'index_*.dawg')
    wikidump = find(root, '*-pages-articles.xml')
    
    # Get disambiguations
    if not any(root.glob('disambig.json')):
        wikidata_disambigfile = find(root, 'wikidata*-disambig.txt')
        get_disambig(wikidump, dawgfile, wikidata_disambigfile, nparts=nparts)
    disambigfile = find(root, 'disambig.json')
    
    # Get paragraph links
    if not any(root.glob('*-paragraph-links')):
        get_paragraphs(wikidump, dawgfile)
    paragraphlinks = find(root, '*-paragraph-links')
    
    # Count
    curdir = root
    stem = (True, False) if (stem is None) else (stem,)
    for p_stem in stem: # parameter sweep
        for p_min_count in min_count: # parameter sweep
            newdir = curdir / f'count--{"" if p_stem else "no-"}stem--min_count={p_min_count}'
            newdir.mkdir(parents=True, exist_ok=True)
            print(newdir)
            # count(paragraphlinks, min_count = p_min_count, stem = p_stem, head=head)
    