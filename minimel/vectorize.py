"""
Vectorize paragraph text dataset
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging
import regex as re
import html
import pickle
import json

import scipy.sparse
import tqdm
import pandas as pd
import numpy as np

from .normalize import normalize

token_pattern = re.compile(r"(?u)\b\w\w+\b")
    
def enc(x, dim=2**18):
    return 'f'+str( abs(hash(x)) % dim )
    
def vw_tok(text, dim=2**18):
    # return [enc(t, dim=dim) for t in token_pattern.findall(text.lower())]
    return [t for t in token_pattern.findall(text.lower()) if ('|' not in t) and (':' not in t)]

def vw(lines, anchor_json: pathlib.Path, dim=2**18, balanced=False, use_ns=True, ldf=False):
    surface_weights = json.load(anchor_json.open())
    surface_weights = {
        m:{int(e.replace('Q','')):c for e,c in ec.items()} 
        for m, ec in surface_weights.items()
    }
    ents = set(e for ec in surface_weights.values() for e in ec)
    e_i = {e:i for i,e in enumerate(sorted(ents))}
    
    outlines = []
    for line in lines:
        pgid, mention_ent, text = line.split('\t', 2)
        mention_ent = json.loads(mention_ent)
        tokens = vw_tok(text, dim=dim)
        if ldf:
            ldf_ok = False
        for m, e in mention_ent.items():
            for norm in normalize(m):
                if norm in surface_weights and e in surface_weights[norm]:
                    weights = surface_weights[norm]
                    if len(weights) > 1:
                        if ldf and not ldf_ok:
                            outlines.append(f'shared | ' + ' '.join(tokens))
                            ldf_ok = True
                        labels = [f'{e_i[e]}:0']
                        for o,c in weights.items():
                            if o != e:
                                w = int(np.log1p(c)) if balanced else 1
                                labels += [f'{e_i[o]}:{w}']
                        if labels and tokens:
                            if ldf:
                                ns = '_'.join(vw_tok(norm))
                                for l in labels:
                                    outlines.append(f'{l} | {ns}')
                            elif use_ns:
                                ns = '_'.join(vw_tok(norm))
                                if ns:
                                    out = ' '.join(labels) + f' |{ns} ' + ' '.join(tokens)
                                    outlines.append(out)
                            else:
                                out = ' '.join(labels) + ' | ' + ' '.join(tokens)
                                outlines.append(out)
        if ldf and ldf_ok:
            outlines.append('')
    outlines.append('')
    return outlines


class TransLiterator:
    def __init__(self, lang):
        import requests
        url = f'https://raw.githubusercontent.com/snowballstem/snowball/master/algorithms/{lang}.sbl'
        resp = requests.get(url)
        if resp.ok:
            defs = [l[9:].split(None, 1) for l in resp.text.splitlines() if l.startswith('stringdef')]            
            self.charmap = {f'\\u{code[4:-2]}'.encode().decode('unicode_escape'):name for name, code in defs}
    
    def code(self, text):
        for a,b in self.charmap.items():
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

    m = fasttext.load_model(str(embeddingsfile))
    return np.vstack([m.get_sentence_vector(p)[:dim] for p in paragraphs])


def filter_paragraphs(lines, anchor_file):
    anchor_scores = json.load(open(anchor_file))
    surface_num = {l: i for i, l in enumerate(anchor_scores)}

    output = []
    for line in lines:
        page, links, paragraph = line.strip().split("\t", 2)
        links = json.loads(links)
        links = {n: i for l, i in links.items() for n in normalize(l)}
        if any(l in anchor_scores for l in links):
            links = {surface_num[l]: i for l, i in links.items() if l in surface_num}
            if paragraph:
                output.append((links, paragraph))
    return output


def cull_empty_partitions(df):
    import dask.dataframe as dd

    ll = list(df.map_partitions(len).compute())
    df_delayed = df.to_delayed()
    df_delayed_new = list()
    pempty = None
    for ix, n in enumerate(ll):
        if 0 == n:
            pempty = df.get_partition(ix)
        else:
            df_delayed_new.append(df_delayed[ix])
    if pempty is not None:
        df = dd.from_delayed(df_delayed_new, meta=pempty)
    return df

def vectorize(
    paragraphlinks: pathlib.Path,
    anchor_json: pathlib.Path,
    *,
    vectorizer: pathlib.Path = None,
    dim: int = None,
    lang: str = None,
    balanced: bool = False,
    ns: bool = True,
    ldf: bool = False,
):
    """
    Vectorize paragraph text dataset into Vowpal Wabbit format

    Args:
        paragraphlinks: Paragraph links directory
        anchor_json: Anchor count json file
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use HashingVectorizer.
        lang: ICU tokenizer language
        balanced: balanced training
        use_ns: Use surface form as 
        ldf: ldf
    """
                
    if vectorizer:
        name = anchor_json.stem + "." + vectorizer.stem
    else:
        name = anchor_json.stem
    
    b = '.bal' if balanced else ''
    n = '.nons' if not ns else ''
    l = '.ldf' if ldf else ''
    fname = (anchor_json.parent / f'{name}{b}{n}{l}.parts')
    logging.info(f"Writing to {fname}")
    
    import dask.bag as db
    from .scale import progress, get_client
    
    # if language:
    #     logging.info(f"Snowball stemming for language: {language}")

    with get_client():

        bag = db.read_text(str(paragraphlinks) + "/*", files_per_partition=3)
        data = (
            bag.map_partitions(
                vw, anchor_json, 
                dim=(dim or 2**18), 
                balanced=balanced,
                use_ns = ns,
                ldf = ldf,
            )#, language=language)
            .to_textfiles(str(fname))
        )
        # with open(fname, 'w') as fw:
        #     for line in data:
        #         print(line, file=fw)
        return fname
