from .index import index
from .get_disambig import get_disambig, query_pages
from .get_paragraphs import get_paragraphs

import pathlib, logging


def prepare(
    wikiname: str,
    version: str,
    *,
    rootdir: pathlib.Path = None,
    mirror: str = "https://dumps.wikimedia.org",
    overwrite: bool = False,
    nparts: int = 100,
    index_only: bool = False,
    custom_langcode: str = None,
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
        index_only: Whether to only create the DAWG index
        custom_langcode: Custom language code (if different from wikiname, e.g. "en-simple")
    """
    from shutil import which

    assert which("bunzip2"), "bunzip2 not found!"
    import subprocess

    from wikimapper import download_wikidumps, create_index
    from wikimapper.download import _download_file

    rootdir = rootdir or pathlib.Path.cwd()
    dumpname = f"{wikiname}-{version}"
    dawgfile = rootdir / f"index_{dumpname}.dawg"
    if not overwrite and not dawgfile.exists():
        logging.info("Downloading & creating entity index...")
        download_wikidumps(dumpname, rootdir, mirror, overwrite)
        index_fname = rootdir / f"index_{dumpname}.db"
        if not overwrite and not index_fname.exists():
            create_index(dumpname, rootdir, index_fname)
        index(index_fname)

    if not index_only:
        pl_dir = rootdir / f"{dumpname}-paragraph-links"
        if not overwrite and not pl_dir.exists():
            logging.info("Downloading wikipedia dump...")
            pa_fname = f"{dumpname}-pages-articles.xml"
            pa_url = mirror + f"/{wikiname}/{version}/" + pa_fname + ".bz2"
            wikidump = rootdir / pa_fname
            if not overwrite and not wikidump.exists():
                pa_zip = rootdir / (pa_fname + ".bz2")
                _download_file(pa_url, pa_zip, overwrite)
                subprocess.run(["bunzip2", pa_zip])

            logging.info("Extracting paragraphs from wikipedia dump...")
            get_paragraphs(wikidump, dawgfile, nparts=nparts)

        lang = wikiname.split("wiki")[0]
        disambigpages = rootdir / "ents-disambig.txt"
        logging.info("Querying Wikidata for disambiguation pages...")
        if not overwrite and not disambigpages.exists():
            query_pages(custom_langcode or lang, outfile=disambigpages)

        logging.info("Extracting disambiguation pages...")
        get_disambig(wikidump, dawgfile, disambigpages, nparts=nparts)
