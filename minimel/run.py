import typing
import pathlib
import pickle
import json
import sys
import logging
from contextlib import redirect_stdout

import dawg
import tqdm
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

from .normalize import normalize
from .vectorize import hashvec, transform, embed, vectorize
from .train import train

def vectorize_text(texts, vectorizer=None, dim=None):
    if not vectorizer:
        return hashvec(texts, dim=dim)
    else:
        assert pathlib.Path(vectorizer).exists()
        try:
            pickle.load(open(vectorizer, "rb"))  # only check
            return transform(texts, vectorizer)
        except pickle.UnpicklingError:
            return embed(texts, vectorizer)


def make_lr(params, dim=None):
    lr = LogisticRegression()
    lr.classes_ = np.array(params.reset_index()["class"])
    lr.intercept_ = params.intercept.values
    issparse = lambda x: any(type(v) == list for v in x or ())
    sparse = params.coef.map(issparse).any()

    def coefs(params):
        for col, ccoef in enumerate(list(params.coef)):
            if ccoef:
                for row, data in enumerate(ccoef):
                    if type(data) == list:
                        row, data = data
                    yield data, (row, col)

    data, rowcol = zip(*coefs(params))
    row, col = zip(*rowcol)
    shape = (params.shape[0], dim or 1048576) if dim or sparse else None
    lr.coef_ = scipy.sparse.csr_matrix((data, (col, row)), shape=shape)
    return lr

def get_scores(golds, preds):
    gold, pred = zip(
        *(
            ((gs or {}).get(surface, -1) or -1, (ps or {}).get(surface, -1) or -1)
            for (gs, ps) in zip(golds, preds)
            for surface in set(gs or {}) | set(ps or {})
        )
    )
    res = pd.DataFrame(
        {
            avg: precision_recall_fscore_support(
                gold, pred, zero_division=0, average=avg
            )[:-1]
            for avg in ["micro", "macro"]
        },
        index=["precision", "recall", "fscore"],
    )
    res = res.unstack().T
    res.loc[("", "support")] = len(gold)
    return res

def run(
    dawgfile: pathlib.Path,
    modelfile: pathlib.Path = None,
    *infile: pathlib.Path,
    vectorizer: pathlib.Path = None,
    dim: int = None,
    lang: str = None,
    countfile: pathlib.Path = None,
    evaluate: bool = False,
    only_predictions: bool = False,
):
    """
    Perform entity linking

    Args:
        dawgfile: DAWG trie file of Wikipedia > Wikidata count
        modelfile: Parquet file of Logistic Regression coefficients
        infile: Input file (- or absent for standard input). TSV rows of
            (ID, {surface -> ID}, text) or ({surface -> ID}, text) or (text)

    Keyword Arguments:
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use HashingVectorizer.
        dim: Dimensionality cutoff. If using HashingVectorizer, hash dimensions.
        countfile: Additional preferred deterministic surfaceform -> ID json
        evaluate: Report evaluation scores instead of predictions
        only_predictions: Only print predictions, not original text
    """
    if (not infile) or (infile == "-"):
        infile = (sys.stdin, )
    logging.debug(f'Reading from {infile}')

    models = {}
    if modelfile:
        gs = pd.read_parquet(str(modelfile)).groupby(level=0)
        it = tqdm.tqdm(gs, "Loading models") if logging.root.level < 30 else gs
        for surface, params in it:
            models[surface] = make_lr(params, dim)

    index = dawg.IntDAWG()
    index.load(str(dawgfile))

    count = json.load(countfile.open()) if countfile else {}

    ids, ents, texts = (), (), ()
    data = pd.concat([pd.read_csv(i, sep="\t", header=None) for i in infile])
    if data.shape[1] == 1:
        texts = data[0]
    elif data.shape[1] == 2:
        ids, texts = data[0], data[1]
    elif data.shape[1] == 3:
        data[1] = data[1].map(json.loads)
        ids, ents, texts = data[0], data[1], data[2]

    if models:
        it = tqdm.tqdm(texts, "Vectorizing") if logging.root.level < 30 else texts
        X = vectorize_text(it, vectorizer=vectorizer, dim=dim)
        logging.info(f"Vectorized data of shape {X.shape}")

    preds = []
    it = tqdm.tqdm(texts, "Predicting") if logging.root.level < 30 else texts
    for i, text in enumerate(it):
        ent_pred = {}
        if len(ents):
            for surface in ents[i]:
                pred = None
                for norm in normalize(surface, language=lang):
                    m = models.get(norm, None)
                    if m:
                        try:
                            pred = int(m.predict(X[i].reshape(1, -1))[0])
                        except ValueError:
                            logging.error(f"Model coef is {m.coef_.shape}")
                            raise
                    elif norm in count:
                        dist = count[norm]
                        pred = max(dist, key=lambda x: dist[x])
                    elif surface.replace(" ", "_") in index:
                        pred = index[surface.replace(" ", "_")]
                    if pred:
                        ent_pred[surface] = int(str(pred).replace('Q',''))
        if not evaluate:
            if only_predictions:
                if len(ids):
                    print(ids[i], json.dumps(ent_pred), sep="\t")
                else:
                    print(json.dumps(ent_pred), sep="\t")
            else:
                if len(ids):
                    print(ids[i], json.dumps(ent_pred), text, sep="\t")
                else:
                    print(json.dumps(ent_pred), text, sep="\t")
            
        preds.append(ent_pred)

    if len(ents) and evaluate:
        print(get_scores(ents, preds).T.to_csv())

def evaluate(
    goldfile: pathlib.Path,
    *predfiles: pathlib.Path,
    agg: typing.List[pathlib.Path] = (),
):
    """
    Evaluate predictions
    
    Args:
        gold: Gold data TSV
        pred: Prediction TSVs
    
    Keyword Arguments:
        agg: Aggregation jsons (TODO: depend on data...?)
    
    """
    data = pd.read_csv(str(goldfile), sep="\t", header=None)
    assert data.shape[1] > 1
    gold = data[0] if data.shape[1] == 2 else data.set_index(0)[1]
    gold = gold.map(json.loads)
        
    scores = {}
    for predfile in tqdm.tqdm(predfiles, 'Evaluating'):
        data = pd.read_csv(str(predfile), sep="\t", header=None)
        pred = data[0] if data.shape[1] == 1 else data.set_index(0)[1]
        pred = pred.map(json.loads)
        preddf = pd.DataFrame({'gold':gold, 'pred':pred })
        scores[predfile.stem] = get_scores(preddf.gold, preddf.pred)
        
    pd.set_option('display.max_colwidth', None)
    print(pd.DataFrame(scores).T)
    
def experiment(
    filtered_counts_file: pathlib.Path,
    infile: pathlib.Path,
    root: pathlib.Path = pathlib.Path('.'),
    *,
    vectorizers: typing.List[pathlib.Path] = (),
    stem: str = None,
    maxtrain:int = 1000,
):
    """
    Run experiment
    """
    dawgfile = (root / f'index_{root.resolve().stem}.dawg')
    countfile = (root / f'{filtered_counts_file.stem.rsplit(".", 1)[0]}.json')
    logging.info(f'Using count file {countfile}')
    
    paragraphlinks_dir = (root / f'{root.resolve().stem}-paragraph-links')
    assert paragraphlinks_dir.exists(), paragraphlinks_dir
    assert filtered_counts_file.exists(), filtered_counts_file
    
    predfiles = []
    models_vectorizers = [(None,None)]
    filt = filtered_counts_file.stem
    ext = 'logreg.parquet'
    
    hash_filestem = f'{filt}.hash1048576'
    for unbal in [False, True]:
        u = '.unbal' if unbal else ''
        hashmodelfile = (root / f'{hash_filestem}.max{maxtrain}{u}.{ext}')
        if not any(root.glob(hash_filestem + '*')):
            vectorize(paragraphlinks_dir, filtered_counts_file)
        if not hashmodelfile.exists():
            train(filtered_counts_file, (root / hash_filestem), max_samples=maxtrain, unbalanced=unbal)
        models_vectorizers.append( (hashmodelfile, None) )

    for vec in vectorizers:
        vec_filestem = f'{filt}.{vec.stem}'
        for unbal in [False, True]:
            u = '.unbal' if unbal else ''
            vecmodelfile = (root / f'{vec_filestem}.max{maxtrain}{u}.{ext}')
            if not any(root.glob(vec_filestem + '*')):
                vectorize(paragraphlinks_dir, filtered_counts_file, vectorizer=vec)
            if not vecmodelfile.exists():
                train(filtered_counts_file, (root / vec_filestem), max_samples=maxtrain, unbalanced=unbal)
            models_vectorizers.append( (vecmodelfile, vec) )

    for count_arg in (None, countfile):
        for modelfile, vectorizer in models_vectorizers:
            outfile = 'pred-mewsli-'
            if modelfile:
                outfile += modelfile.stem.rsplit('.', 1)[0]
            else:
                outfile += 'base'
            outfile += ('-' + count_arg.stem) if count_arg else ''
            outfile += '.tsv'
            predfiles.append( pathlib.Path(outfile) )
            logging.info(f'Writing to {outfile}')
            with open(outfile, 'w') as f, redirect_stdout(f):
                run(
                    dawgfile,
                    modelfile,
                    infile,
                    vectorizer=vectorizer,
                    lang=stem,
                    countfile=count_arg,
                    only_predictions=True,
                )
    
    evaluate(
        infile,
        *sorted(predfiles),
    )