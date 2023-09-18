"""
Convert Wikimapper index to IntDAWG.
"""
import sqlite3, pathlib, logging, sys

import tqdm

from .scale import fileparts


def make_dawg(db_fname):
    import dawg

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


def xml_db(wikidump: pathlib.Path, *, ns: int = 0, nparts: int = 100):
    """
    Make a name database from Wikidump page ids

    Args:
        wikidump: Wikipedia XML dump file

    Keyword Arguments:
        ns: Page Namespace
        nparts: Number of chunks to read
    """
    import sqlite3
    import xml.etree.cElementTree as cElementTree
    import dask.bag as db
    from .scale import progress, get_client

    def get_ids(lines, ns: int):
        output = []
        for line in lines:
            line = line.replace("\\n", "\n")
            elem = cElementTree.fromstring(line)
            if int(elem.find("./ns").text) == ns:
                title = elem.find("./title").text.replace(" ", "_")
                id = int(elem.find("./id").text)
                output.append({"title": title, "id": id})
        return output

    with get_client():
        bag = db.from_sequence(range(nparts), npartitions=nparts).map_partitions(
            fileparts, wikidump, nparts, "<page>", "</page>"
        )

        page_id = bag.map_partitions(get_ids, ns)
        if logging.root.level < 30:
            progress(page_id.persist(), out=sys.stderr)
        else:
            page_id.persist()

        data = list(page_id.compute())
        outfile = wikidump.parent / f"index_{wikidump.stem}.db"
        logging.info(f"Saving to {outfile}...")
        con = sqlite3.connect(str(outfile))
        cur = con.execute(
            "CREATE TABLE mapping(wikipedia_id, wikipedia_title, wikidata_id)"
        )
        cur.executemany("INSERT INTO mapping VALUES(:id, :title, :id)", data)
        n = cur.execute("SELECT COUNT(*) FROM mapping").fetchall()
        logging.info(f"Saved {n} entries")
        cur.close()
        con.close()


if __name__ == "__main__":
    import defopt

    defopt.run([index, xml_db])
