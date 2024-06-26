from .normalize import normalize, SNOWBALL_LANG as code_name
from .index import index, xml_db
from .get_disambig import get_disambig, query_pages
from .get_paragraphs import get_paragraphs
from .count import count
from .mentions import count_names
from .clean import clean
from .ent_feats import ent_feats
from .vectorize import vectorize
from .train import train
from .run import run, evaluate, MiniNED
from .audit import audit
from .experiment import experiment

import pathlib, logging

def prepare(
    wikiname: str,
    version: str,
    *,
    rootdir: pathlib.Path = None,
    mirror: str = "https://dumps.wikimedia.org",
    overwrite: bool = False,
    nparts: int = 100,
):
    """
    Download required files and make indices

    Args:
        wikiname: Wikipedia edition name (eg. "simplewiki")
        version: Wikipedia version (eg. "latest")

    Keyword arguments:
        rootdir: Root directory
        mirror: Wikimedia mirror
        overwrite: Whether to overwrite existing files
        nparts: Number of chunks to read
    """
    from shutil import which
    assert which('bunzip2'), 'bunzip2 not found!'
    import subprocess

    from wikimapper import download_wikidumps, create_index
    from wikimapper.download import _download_file

    logging.info("Downloading & creating entity index...")
    rootdir = rootdir or pathlib.Path.cwd()
    dumpname = f'{wikiname}-{version}'
    download_wikidumps(dumpname, rootdir, mirror, overwrite)
    create_index(dumpname, rootdir, None)

    db_fname = rootdir / f'index_{dumpname}.db'
    index(db_fname)

    logging.info("Downloading wikipedia dump...")
    pa_fname = f'{dumpname}-pages-articles.xml'
    pa_url = mirror + f'/{wikiname}/{version}/' + pa_fname + '.bz2'
    _download_file(pa_url, rootdir / (pa_fname + '.bz2'), overwrite)
    subprocess.run(["bunzip2", rootdir / (pa_fname + '.bz2')])

    logging.info("Extracting paragraphs from wikipedia dump...")
    wikidump = rootdir / pa_fname
    dawgfile = rootdir / f'index_{dumpname}.dawg'
    get_paragraphs(wikidump, dawgfile, nparts=nparts)

    lang = wikiname.split('wiki')[0]
    disambigpages = rootdir / 'ents-disambig.txt'
    logging.info("Querying Wikidata for disambiguation pages...")
    query_pages(lang, outfile=disambigpages)

    logging.info("Extracting disambiguation pages...")
    get_disambig(wikidump, dawgfile, disambigpages, nparts=nparts)
