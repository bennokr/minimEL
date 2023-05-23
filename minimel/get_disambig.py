"""
Extract list-hyperlinks from Wikipedia disambiguation pages
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import sys, pathlib, argparse, logging, json
import xml.etree.cElementTree as cElementTree

import mwparserfromhell

try:
    import dawg
except ImportError:
    import dawg_python as dawg

from .scale import fileparts


def get_list_links(page, disambig_template=None):
    code = mwparserfromhell.parse(page)
    nodes = code.filter(matches=(lambda x: str(x).strip()), recursive=True)

    if disambig_template:
        ts = code.ifilter_templates()
        if not any(t.name.lower() == disambig_template.lower() for t in ts):
            return

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


def get_disambig_links(
    lines, dawgfile, wikidata_disambigfile=None, disambig_template=None
):
    index = dawg.IntDAWG()
    index.load(dawgfile)

    if wikidata_disambigfile:
        disambig_ents = set(
            int(q.replace("Q", "")) for q in open(wikidata_disambigfile).readlines()
        )
    output = []
    for line in lines:
        line = line.replace("\\n", "\n")
        elem = cElementTree.fromstring(line)
        title = elem.find("./title").text.replace(" ", "_")
        text = elem.find("./revision/text").text

        if (not wikidata_disambigfile) or (index.get(title) in disambig_ents):
            links = set()
            for link in get_list_links(text, disambig_template=disambig_template):
                link = link.replace(" ", "_")
                if link in index:
                    links.add(index[link])
            if links:
                output.append((title, list(links)))
    return output


def get_disambig(
    wikidump: pathlib.Path,
    dawgfile: pathlib.Path,
    wikidata_disambigfile: pathlib.Path = None,
    *,
    disambig_template: str = None,
    nparts: int = 1000,
):
    """
    Get disambiguation links.

    Writes `disambig.json`.

    Args:
        wikidump: Wikipedia XML dump file
        dawgfile: DAWG trie file of Wikipedia > Wikidata mapping
        wikidata_disambigfile: Flat text file of disambiguation pages with one Wikidata Q.. per line

    Keyword Arguments:
        nparts: Number of chunks to read
        disambig-template: If no wikidata_disambigfile is provided, consider disambiguation pages that contain a template with this name
    """

    import dask.bag as db
    from .scale import progress, get_client

    if disambig_template:
        logging.info(
            f"Using disambiguation template {disambig_template}, not {wikidata_disambigfile}"
        )

    with get_client():
        bag = db.from_sequence(range(nparts), npartitions=nparts).map_partitions(
            fileparts, wikidump, nparts, "<page>", "</page>"
        )

        links = bag.map_partitions(
            lambda b: get_disambig_links(
                b,
                str(dawgfile),
                str(wikidata_disambigfile or ""),
                disambig_template=disambig_template,
            )
        )
        logging.info(f"Extracting disambiguation links...")
        if logging.root.level < 30:
            progress(links.persist(), out=sys.stderr)
        else:
            links.persist()

        links = dict(links.compute())
        outfile = wikidump.parent / "disambig.json"
        logging.info(f"Writing to {outfile}")
        with open(outfile, "w") as fw:
            json.dump(links, fw)
