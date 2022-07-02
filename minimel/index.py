"""
Convert Wikimapper index to IntDAWG.
"""
import sqlite3, dawg, tqdm, pathlib, argparse, logging


def make_dawg(db_fname):
    db = sqlite3.connect(db_fname)
    title_id = {}
    rows = db.execute("select * from mapping")
    if logging.INFO >= logging.root.level:
        (total,) = next(db.execute("select count(*) from mapping"))
        rows = tqdm.tqdm(rows, total=total, desc="Loading mapping...")
    for pageid, pagetitle, wikidata_id in rows:
        if pagetitle == "\x00":
            continue
        if pagetitle and wikidata_id:
            title_id[pagetitle] = int(wikidata_id[1:])

    logging.info("Building IntDAWG trie...")
    return dawg.IntDAWG(title_id.items())


def index(db_fname: pathlib.Path):
    """
    Make an efficient DAWG trie index from a Wikimapper sqlite file

    Args:
        db_fname: Wikimapper SQLite3 index file
    """
    d = make_dawg(str(db_fname))
    outfile = db_fname.parent / f"{db_fname.stem}.dawg"
    logging.info(f"Saving to {outfile}...")
    d.save(str(outfile))


if __name__ == "__main__":
    import defopt

    defopt.run(index)
