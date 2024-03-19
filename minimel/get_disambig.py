"""
Extract list-hyperlinks from Wikipedia disambiguation pages
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, logging, json
import xml.etree.cElementTree as cElementTree
import contextlib, sys


def writer(fn): 
    @contextlib.contextmanager
    def stdout():
        yield sys.stdout
    return open(fn, 'w') if fn else stdout()

def query_pages(langcode: str, *, query_listpages: bool = False, outfile: pathlib.Path = None):
    """
    Query the Wikidata API to get disambiguation (& list pages if indicated)

    Returns Wikidata Qids, one per line

    Args:
        langcode: Wikipedia language code

    Keyword Arguments:
        query_listpages: Whether to also query for list pages

    """
    import requests

    listquery = """UNION
        {?s wdt:P31 wd:Q13406463 .} # list
        UNION
        {?s wdt:P360 ?l . } # list of
    """ if query_listpages else ''

    query = """
        SELECT DISTINCT ?s WHERE {
        {?s wdt:P31 wd:Q4167410 .} # disambig
        %s

        ?page schema:about ?s .
        ?page schema:inLanguage "%s" .
        }
        """ % (listquery, langcode)
    url = 'https://query.wikidata.org/sparql'
    params = {'format': 'json', 'query': query}
    results = requests.get(url, params=params).json()
    bindings = results.get('results', []).get('bindings', [])
    qids = [b.get('s', {}).get('value', '')[31:] for b in bindings]

    logging.info(f"Writing to {outfile}")
    with writer(outfile) as fw:
        for q in qids:
            print(q, file=fw)


def get_list_links(page, disambig_template=None):
    import mwparserfromhell

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


def get_disambig_links(lines, dawgfile, disambig_ent_file=None, disambig_template=None):
    try:
        import dawg
    except ImportError:
        import dawg_python as dawg
        
    index = dawg.IntDAWG()
    index.load(dawgfile)

    if not disambig_template:
        disambig_ents = set(
            int(q.replace("Q", "")) for q in open(disambig_ent_file).readlines()
        )
    output = []
    for line in lines:
        line = line.replace("\\n", "\n")
        elem = cElementTree.fromstring(line)
        title = elem.find("./title").text.replace(" ", "_")
        text = elem.find("./revision/text").text

        if disambig_template:
            ent = int(elem.find("./id").text)
        else:
            ent = index.get(title)

        if disambig_template or (ent in disambig_ents):
            links = set()
            for link in get_list_links(text, disambig_template=disambig_template):
                link = link.replace(" ", "_")
                if link in index:
                    links.add(index[link])
            if links:
                output.append((ent, title, list(links)))
    return output


def get_disambig(
    wikidump: pathlib.Path,
    dawgfile: pathlib.Path,
    disambig_ent_file: pathlib.Path = None,
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
        disambig_ent_file: Flat text file of disambiguation pages with one entity ID per line

    Keyword Arguments:
        nparts: Number of chunks to read
        disambig_template: Use disambiguation pages that contain a template with this name instead of `disambig_ent_file` (if disambig_ent_file is provided, create it)
    """

    import dask.bag as db
    from .scale import progress, get_client, fileparts

    if disambig_template:
        logging.info(
            f"Using disambiguation template {disambig_template}, not {disambig_ent_file}"
        )

    with get_client():
        bag = db.from_sequence(range(nparts), npartitions=nparts).map_partitions(
            fileparts, wikidump, nparts, "<page>", "</page>"
        )

        links = bag.map_partitions(
            lambda b: get_disambig_links(
                b,
                str(dawgfile),
                str(disambig_ent_file or ""),
                disambig_template=disambig_template,
            )
        )
        logging.info(f"Extracting disambiguation links...")
        if logging.root.level < 30:
            progress(links.persist(), out=sys.stderr)
        else:
            links.persist()

        ents, titles, links = zip(*links.compute())
        if disambig_template and disambig_ent_file:
            logging.info(f"Writing to {disambig_ent_file}")
            with open(disambig_ent_file, "w") as fw:
                for e in ents:
                    print(e, file=fw)
        links = dict(zip(titles, links))
        outfile = wikidump.parent / "disambig.json"
        logging.info(f"Writing to {outfile}")
        with open(outfile, "w") as fw:
            json.dump(links, fw)
