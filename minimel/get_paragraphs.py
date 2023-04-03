"""
Extract hyperlinks from Wikipedia dumps
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging, re
import os, re, codecs
import xml.etree.cElementTree as cElementTree

import dawg
import mwparserfromhell
import mwparserfromhell.nodes as nodes

from .scale import fileparts

BADSTART = ["{{", "[", "|"]  # TODO: filter out paragraphs that are only links


def get_str(node):
    if type(node) == nodes.wikilink.Wikilink:
        s = str(node.text or node.title)
        if "|" not in s:
            return s
    if type(node) == nodes.text.Text:
        return str(node)
    return ""


good = [nodes.text.Text, nodes.tag.Tag, nodes.wikilink.Wikilink]


def get_text(w):
    text = ""
    for p in w.ifilter(matches=lambda x: type(x) in good, recursive=False):
        if type(p) == nodes.tag.Tag:
            if p.wiki_markup and p.contents:
                for n in p.contents.nodes:
                    text += get_str(n)
        else:
            text += get_str(p)
    return text.replace("\n", " ").strip()


def get_links(w, index):
    for l in w.ifilter_wikilinks(recursive=True):
        t = str(l.title)
        if t and not re.match("^[A-Z][a-z]+:", t):
            t = t[0].upper() + (t[1:] if len(t) > 1 else "")
            t = t.replace(" ", "_")
            if t in index:
                yield str(l.text or l.title), index[t]


def process_line(pagename, mwcode, index, skip=None):
    skip = list(skip) or []
    if (not mwcode) or mwcode.startswith("#"):
        return
    pagelabel = pagename.replace("_", " ").split(" (")[0]
    pageids = set([index[pagename]]) if pagename in index else set()
    # Keep track of (label, wikidata-ID) pairs
    all_links = set([(pagelabel, i) for i in pageids])
    paragraphs = mwcode.split("\n\n")
    for paragraph in paragraphs:
        w = mwparserfromhell.parse(paragraph)
        links, text = set(get_links(w, index)), get_text(w)
        if text and not any(text.startswith(b) for b in BADSTART + skip):
            # Enrich: add links long-to-short, non-overlapping
            for s, e in sorted(all_links, key=lambda x: len(x[0])):
                if (s in text) and not any(s in l for l, _ in links):
                    links.add((s, e))
            all_links |= links
            if links:
                yield pagename, links, w, text


def get_anchor_paragraphs(lines, dawgfile, skip=[]):
    import dawg, json

    index = dawg.IntDAWG()
    index.load(str(dawgfile))
    output = []
    for line in lines:
        elem = cElementTree.fromstring(line)
        title = elem.find("./title").text.replace(" ", "_")
        text = elem.find("./revision/text").text
        for name, links, code, text in process_line(title, text, index, skip=skip):
            output.append((name, json.dumps(dict(links)), text))
    return output


def get_paragraphs(
    wikidump: pathlib.Path, dawgfile: pathlib.Path, *skip: str, nparts: int = 1000
):
    """
    Extract hyperlinks from Wikipedia dumps.

    Writes to `outdir`.

    Args:
        wikidump: Wikipedia pages-articles XML dump file
        dawgfile: DAWG trie file of Wikipedia > Wikidata mapping
        skip: Skip pages with this prefix

    Keyword Arguments:
        nparts: Number of chunks to read
    """
    import dask.bag as db
    from .scale import progress, get_client

    with get_client():

        bag = db.from_sequence(range(nparts), npartitions=nparts).map_partitions(
            fileparts, wikidump, nparts, "<page>", "</page>"
        )

        anchors = bag.map_partitions(lambda b: get_anchor_paragraphs(b, dawgfile, skip))

        stem = str(wikidump.stem).replace("pages-articles", "paragraph-links")
        outglob = str(wikidump.parent) + "/" + stem + "/*.tsv"
        tasks = anchors.map("\t".join).to_textfiles(outglob)
