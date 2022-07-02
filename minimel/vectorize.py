"""
Vectorize paragraph text dataset
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging
import re, html
import pickle
import scipy.sparse

import json
import pandas as pd
import numpy as np

from .normalize import normalize


def hashvec(paragraphs, dim=None, lang=None, tokenizer=None):
    from sklearn.feature_extraction.text import HashingVectorizer

    if lang:
        from icu_tokenizer import Tokenizer

        tokenizer = Tokenizer(lang=lang).tokenize
    vec = HashingVectorizer(
        n_features=(dim or 1048576),
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
    Vectorize paragraph text dataset

    Args:
        paragraphlinks: Paragraph links directory
        anchor_json: Anchor count json file
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use HashingVectorizer.
        dim: Dimensionality cutoff. If using HashingVectorizer, hash dimensions.
        lang: ICU tokenizer language
    """
    import dask.bag as db
    from .scale import progress, get_client

    with get_client():
        inglob = str(paragraphlinks) + "/*"
        bag = db.read_text(inglob)
        bag = bag.map_partitions(filter_paragraphs, anchor_json)
        df = bag.to_dataframe(meta=[("links", "O"), ("paragraph", "O")])

        logging.info("Filtering paragraphs")
        progress(df.persist())

        # Get feature array
        df = cull_empty_partitions(df)
        para = df["paragraph"]
        logging.info(f"Featurizing {para.count().compute()} paragraphs")
        if vectorizer is None:
            arr = para.map_partitions(hashvec, dim).persist()
        else:
            assert pathlib.Path(vectorizer).exists()
            try:
                pickle.load(open(vectorizer, "rb"))  # only check
                arr = para.map_partitions(transform, vectorizer).persist()
            except pickle.UnpicklingError:
                arr = para.map_partitions(embed, vectorizer).persist()

        progress(arr)
        arr = arr.compute()
        logging.info(f"Created {arr.shape} {type(arr).__name__}")

        logging.info("Creating index")
        links = df["links"].compute().reset_index(drop=True)
        by_surface, by_ent = links.map(dict.items).explode().str
        
        # TODO: normalize!

        if vectorizer:
            name = anchor_json.stem + "." + vectorizer.stem
        else:
            name = anchor_json.stem + ".hash" + str(dim or 1048576)

        name = str(anchor_json.parent / name)
        if scipy.sparse.issparse(arr):
            logging.info(f"Writing sparse .npz & .index.parquet to {name}")
            scipy.sparse.save_npz(name + ".npz", arr)
        else:
            logging.info(f"Writing dense .npy & .index.parquet to {name}")
            np.save(name + ".npy", arr.astype(np.float16))

        index_df = pd.DataFrame(dict(by_surface=by_surface, by_ent=by_ent))
        index_df.to_parquet(name + ".index.parquet")
