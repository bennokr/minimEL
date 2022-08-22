"""
Vectorize paragraph text dataset
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging
import re, html
import pickle
import json

import scipy.sparse
import tqdm
import pandas as pd
import numpy as np

from .normalize import normalize

token_pattern = re.compile(r"(?u)\b\w\w+\b")
    
def enc(x, dim=2**18):
    return str( abs(hash(x)) % dim )
    
def vw_tok(text, dim=2**18):
    return [enc(t, dim=dim) for t in token_pattern.findall(text.lower())]

def vw(lines, surface_weights, dim=2**18):
    for line in lines:
        pgid, mention_ent, text = line.split('\t', 2)
        mention_ent = json.loads(mention_ent)
        tokens = vw_tok(text, dim=dim)
        for m, e in mention_ent.items():
            if m in surface_weights:
                if len(surface_weights[m]) > 1:
                    labels = [f'{e}:0']
                    for o in surface_weights[m]:
                        if o != e:
                            labels += [f'{o}:1']
                    if labels and tokens:
                        yield (' '.join(labels), '|', ' '.join(tokens))

                        
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
):
    """
    Vectorize paragraph text dataset into Vowpal Wabbit format

    Args:
        paragraphlinks: Paragraph links directory
        anchor_json: Anchor count json file
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use HashingVectorizer.
        dim: Dimensionality cutoff. If using HashingVectorizer, hash dimensions.
        lang: ICU tokenizer language
    """
    surface_weights = json.load(anchor_json.open())
    surface_weights = {
        m:{int(e.replace('Q','')):c for e,c in ec.items()} 
        for m, ec in surface_weights.items()
    }
                
    if vectorizer:
        name = anchor_json.stem + "." + vectorizer.stem
    else:
        name = anchor_json.stem + ".hash" + str(dim or 2**18)

    fname = (anchor_json.parent / (name + '.dat'))
    with open(fname, 'w') as fw:
        for file in tqdm.tqdm(list(paragraphlinks.glob('*'))):
            # TODO: vectorizer, lang
            for out in vw(file.open(), surface_weights, dim=(dim or 2**18)):
                print(out, file=fw)
    return fname
