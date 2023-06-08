"""
Vectorize paragraph text dataset
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging
import itertools
import shutil
import re
import html
import pickle
import json
import math

try:
    import dawg
except ImportError:
    import dawg_python as dawg
import tqdm

from .normalize import normalize

token_pattern = re.compile(r"(?u)\b\w+\b")


def vw_tok(text):
    return [
        t
        for t in token_pattern.findall(text.lower())
        if ("|" not in t) and (":" not in t)
    ]


def vw(
    lines,
    surface_count_json: pathlib.Path,
    ent_feats_csv=None,
    balanced=False,
    language=False,
    usenil=False,
    head=None,
):
    """Create VW-formatted training data

    Args:
        lines: iterable of (pageid, {name: entityid} json, text) tsv lines
        surface_count_json: path to json file of {name: {entityid: weight}}
        ent_feats_csv: path to csv of (entityid,feat1 feat2 feat3 ...)

    """
    surface_weights = json.load(surface_count_json.open())
    surface_weights = {
        m: {int(e.replace("Q", "")): c for e, c in ec.items()}
        for m, ec in surface_weights.items()
    }
    logging.debug(f"Loaded {len(surface_weights)} surface weights")

    if usenil:
        # Make surfaceform lookup trie
        surface_trie = dawg.CompletionDAWG(surface_weights)

    # Load entity features
    ent_feats = None
    if ent_feats_csv:
        import pandas as pd

        ent_feats = pd.read_csv(ent_feats_csv, header=None, index_col=0, na_values="")
        ent_feats = ent_feats[1].fillna("")
        logging.debug(f"Loaded {len(ent_feats)} entity features")

    def vw_label_lines(weights, e, norm, ent_feats=None):
        if len(weights) > 1:
            entlabels = [(int(e), 0)]  # positive
            for o, c in weights.items():  # negatives
                if o != e:
                    w = int(math.log(1 + c)) if balanced else 1
                    entlabels += [(int(o), w)]
            ns = "_".join(vw_tok(norm))
            for l, w in entlabels:
                efeats = str(ent_feats.get(l, "")) if (ent_feats is not None) else ""
                efeats = "|f " + efeats if efeats else ""
                yield f"{l}:{w} |l {ns}={l} {efeats}"

    outlines = []
    for line in itertools.islice(lines, 0, head):
        pgid, mention_ent, text = line.split("\t", 2)
        try:
            mention_ent = json.loads(mention_ent)
        except ValueError:
            print(mention_ent)
            raise

        tokens = vw_tok(text)  # Tokenize
        if not tokens:
            continue
        labels = []
        for m, e in mention_ent.items():
            for norm in normalize(m, language=language):
                if norm in surface_weights and e in surface_weights[norm]:
                    weights = surface_weights[norm]
                    for label in vw_label_lines(weights, e, norm, ent_feats):
                        labels.append(label)

        if usenil:
            # add NIL data from surface_trie
            for normtext in normalize(text, language=language):
                normtoks = vw_tok(normtext)
                for i, tok in enumerate(vw_tok(normtext)):
                    for comp in surface_trie.keys(tok):
                        if comp not in mention_ent:
                            comp_toks = vw_tok(comp)
                            if tokens[i : i + len(comp_toks)] == comp_toks:
                                # NIL match
                                weights = surface_weights[comp]
                                for label in vw_label_lines(
                                    weights, -1, comp, ent_feats
                                ):
                                    labels.append(label)

        if labels:
            outlines += [f"shared |s " + " ".join(tokens)] + labels + [""]
    outlines.append("")
    logging.info(f"Found {len(outlines)} training instances")
    return outlines


class TransLiterator:
    def __init__(self, lang):
        import requests

        url = f"https://raw.githubusercontent.com/snowballstem/snowball/master/algorithms/{lang}.sbl"
        resp = requests.get(url)
        if resp.ok:
            defs = [
                l[9:].split(None, 1)
                for l in resp.text.splitlines()
                if l.startswith("stringdef")
            ]
            self.charmap = {
                f"\\u{code[4:-2]}".encode().decode("unicode_escape"): name
                for name, code in defs
            }

    def code(self, text):
        for a, b in self.charmap.items():
            text = text.replace(a, b)
        return text


def hashvec(paragraphs, dim=None, lang=None, tokenizer=None):
    from sklearn.feature_extraction.text import HashingVectorizer

    if lang:
        from icu_tokenizer import Tokenizer

        tokenizer = Tokenizer(lang=lang).tokenize
    vec = HashingVectorizer(
        n_features=(dim or 2**18),
        tokenizer=tokenizer,
    )
    return vec.fit_transform(paragraphs)


def transform(paragraphs, vectorizer):
    vec = pickle.load(open(vectorizer, "rb"))
    return vec.transform(paragraphs)


def embed(paragraphs, embeddingsfile, dim=None):
    warnings.simplefilter(action="ignore", category=Warning)
    import fasttext
    import numpy as np

    m = fasttext.load_model(str(embeddingsfile))
    return np.vstack([m.get_sentence_vector(p)[:dim] for p in paragraphs])


def vectorize(
    paragraphlinks: pathlib.Path,
    surface_count_json: pathlib.Path,
    *,
    outfile: pathlib.Path = None,
    head: int = None,
    stem: str = None,
    vectorizer: pathlib.Path = None,
    ent_feats_csv: pathlib.Path = None,
    balanced: bool = False,
    usenil: bool = False,
):
    """
    Vectorize paragraph text dataset into Vowpal Wabbit format

    Args:
        paragraphlinks: Paragraph links directory
        surface_count_json: Surfaceform count json file

    Keyword Arguments:
        outfile: Output file or directory (default: `vec*.parts`)
        head: Use only N first lines from each partition
        stem: Stemming language ISO 639-1 (2-letter) code
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use tokens directly.
        ent_feats_csv: CSV of (ent_id,space separated feat list) entity features
        balanced: Use balanced training
        usenil: Use NIL option
    """

    if vectorizer:
        name = surface_count_json.stem + "." + vectorizer.stem
    else:
        name = surface_count_json.stem

    b = ".bal" if balanced else ""
    f = f".{ent_feats_csv.stem}" if ent_feats_csv else ""
    fname = f"vec.{name}{b}{f}.dat"
    if not outfile:
        outfile = paragraphlinks.parent / fname
    if outfile.is_dir():
        outfile = outfile / fname
    outfile.parent.mkdir(parents=True, exist_ok=True)

    import dask.bag as db
    from .scale import progress, get_client

    from dask.diagnostics import ProgressBar

    if logging.root.level < 30:
        pbar = ProgressBar()
        pbar.register()

    if stem:
        logging.info(f"Snowball stemming for language: {stem}")

    with get_client():
        urlpath = str(paragraphlinks)
        if paragraphlinks.is_dir():
            urlpath += "/*"

        bag = db.read_text(urlpath, files_per_partition=3)
        logging.info(f"Writing to {outfile}.parts")
        data = bag.map_partitions(
            vw,
            surface_count_json,
            head=head,
            ent_feats_csv=ent_feats_csv,
            balanced=balanced,
            language=stem,
            usenil=usenil,
        ).to_textfiles(f"{outfile}.parts")

    logging.info(f"Concatenating to {outfile}")
    with outfile.open("wb") as fout:
        it = pathlib.Path(f"{outfile}.parts").glob("*")
        if logging.root.level < 30:
            it = tqdm.tqdm(list(it), desc="Concatenating")
        for fin in it:
            with fin.open("rb") as f:
                fout.write(f.read())
    shutil.rmtree(str(outfile) + ".parts")

    return outfile
