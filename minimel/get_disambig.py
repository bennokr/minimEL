"""
Extract list-hyperlinks from Wikipedia disambiguation pages
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys, pathlib, argparse, logging, json
import dawg
import xml.etree.cElementTree as cElementTree

import mwparserfromhell

from .scale import fileparts


def get_list_links(page):
    code = mwparserfromhell.parse(page)
    nodes = code.filter(matches=(lambda x: str(x).strip()), recursive=True)
    for i, node in enumerate(nodes[1:]):
        if getattr(nodes[i], "tag", None) == "li":
            if type(node) == mwparserfromhell.nodes.Wikilink:
                yield str(node.title)
            elif hasattr(node, "_contents") and hasattr(
                node._contents, "filter_wikilinks"
            ):
                if node._contents.filter_wikilinks():
                    for link in node._contents.ifilter_wikilinks():
                        yield str(link.title)


def get_disambig_links(lines, dawgfile, disambig_ents_file):
    import dawg

    index = dawg.IntDAWG()
    index.load(dawgfile)

    disambigs = set(
        int(q.replace("Q", "")) for q in open(disambig_ents_file).readlines()
    )
    output = []
    for line in lines:
        line = line.replace("\\n", "\n")
        elem = cElementTree.fromstring(line)
        title = elem.find("./title").text.replace(" ", "_")
        text = elem.find("./revision/text").text

        if index.get(title) in disambigs:
            links = set()
            for link in get_list_links(text):
                link = link.replace(" ", "_")
                if link in index:
                    links.add(index[link])
            if links:
                output.append((title, list(links)))
    return output


def get_disambig(
    wikidump: pathlib.Path,
    dawgfile: pathlib.Path,
    disambigs: pathlib.Path,
    *,
    nparts: int = 1000,
):
    """
    Get disambiguation links.

    Writes `disambig.json`.

    Args:
        wikidump: Wikipedia XML dump file
        dawgfile: DAWG trie file of Wikipedia > Wikidata mapping
        disambigs: Flat text file with Wikidata Q.. per line of disambiguation pages

    Keyword Arguments:
        nparts: Number of chunks to read
    """

    import dask.bag as db
    from .scale import progress, get_client

    with get_client():

        bag = db.from_sequence(range(nparts), npartitions=nparts).map_partitions(
            fileparts, wikidump, nparts, "<page>", "</page>"
        )

        links = bag.map_partitions(
            lambda b: get_disambig_links(b, str(dawgfile), str(disambigs))
        )
        if logging.root.level < 30:
            progress(links.persist(), out=sys.stderr)
        else:
            links.persist()

        links = dict(links.compute())
        outfile = wikidump.parent / "disambig.json"
        logging.info(f"Writing to {outfile}")
        with open(outfile, "w") as fw:
            json.dump(links, fw)
