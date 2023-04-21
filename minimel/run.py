import typing
import pathlib
import pickle
import json
import sys
import glob
import logging
from contextlib import redirect_stdout

import dawg
import tqdm
import pandas as pd
import numpy as np
from vowpalwabbit import pyvw

from .normalize import normalize
from .vectorize import hashvec, transform, embed, vectorize, vw_tok
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


def get_scores(golds, preds):
    from sklearn.metrics import precision_recall_fscore_support

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
    candidatefile: pathlib.Path = None,
    modelfile: pathlib.Path = None,
    *infile: pathlib.Path,
    vectorizer: pathlib.Path = None,
    ent_feats_csv: pathlib.Path = None,
    lang: str = None,
    countfile: pathlib.Path = None,
    evaluate: bool = False,
    predict_only: bool = False,
    score_only: bool = False,
    upperbound: bool = False,
):
    """
    Perform entity linking

    Args:
        dawgfile: DAWG trie file of Wikipedia > Wikidata count
        candidatefile: Candidate {surfaceform -> [ID]} json
        modelfile: Vowpal Wabbit model
        infile: Input file (- or absent for standard input). TSV rows of
            (ID, {surface -> ID}, text) or ({surface -> ID}, text) or (text)

    Keyword Arguments:
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use HashingVectorizer.
        ent_feats_csv: CSV of (ent_id,space separated feat list) entity features
        countfile: Additional preferred deterministic surfaceform -> ID json
        evaluate: Report evaluation scores instead of predictions
        predict_only: Only print predictions, not original text
        upperbound: Create upper bound on performance
    """
    if (not infile) or (infile == "-"):
        infile = (sys.stdin, )
    logging.debug(f'Reading from {infile}')
    
    index = dawg.IntDAWG()
    index.load(str(dawgfile))

    candidates = json.load(candidatefile.open()) if candidatefile else {}
    count = json.load(countfile.open()) if countfile else {}
    
    ent_feats = None
    if ent_feats_csv:
        ent_feats = pd.read_csv(ent_feats_csv, header=None, index_col=0, na_values='')[1]
        ent_feats = ent_feats.fillna('')
        logging.info(f'Loaded {len(ent_feats)} entity features')
    
    model = None
    if candidatefile and modelfile:
        model = pyvw.Workspace(
            initial_regressor=str(modelfile),
            loss_function="logistic",
            csoaa_ldf="mc",
            probabilities=True,
            testonly=True,
            quiet=True,
        )

    ids, ents, texts = (), (), ()
    data = pd.concat([pd.read_csv(i, sep="\t", header=None) for i in infile])
    if data.shape[1] == 1:
        texts = data[0]
    elif data.shape[1] == 2:
        ids, texts = data[0], data[1]
    elif data.shape[1] == 3:
        data[1] = data[1].map(json.loads)
        ids, ents, texts = data[0], data[1], data[2]

    
    def model_predict(model, text, norm, ents, ent_feats=None, score=False):
        # TODO: replace with vectorize.vw!!!
        preds = {}
        ents = list(ents)
        toks = 'shared |s ' + ' '.join(vw_tok(text))
        ns = '_'.join(vw_tok(norm))
        for i,ent in enumerate(ents):
            efeats = str(ent_feats.get(l, '')) if (ent_feats is not None) else ''
            efeats = '|f ' + efeats if efeats else ''
            cands = [f'{e} |l {ns}={e} {efeats}' for e in ents[i:] + ents[:i]]
            preds[ent] = model.predict([toks] + cands)
        if score:
            return preds
        else:
            return max(preds.items(), key=lambda x: x[1])[0]
    
    preds = []
    it = tqdm.tqdm(texts, "Predicting") if logging.root.level < 30 else texts
    for i, text in enumerate(it):
        ent_pred = {}
        if len(ents):
            for surface in ents[i]:
                pred = None
                if upperbound:
                    gold = str(ents[i][surface])
                    for norm in normalize(surface, language=lang):
                        if (norm in count) and (gold in count[norm]):
                            pred = gold
                    if not pred:
                        if str(index.get(surface.replace(" ", "_"), -1)) == gold:
                            pred = gold
                else:
                    for norm in normalize(surface, language=lang):
                        ent_cand = candidates.get(norm, None)
                        if ent_cand and model:
                            pred = model_predict(model, text, norm, ent_cand,
                                                score = score_only)
                        elif norm in count:
                            dist = count[norm]
                            pred = max(dist, key=lambda x: dist[x])
                        elif surface.replace(" ", "_") in index:
                            pred = index[surface.replace(" ", "_")]
                if pred:
                    if score_only:
                        if type(pred) != dict:
                            pred = {pred: 1}
                        pred = {f'Q{p}': s for p,s in pred.items()}
                    else:
                        pred = int(str(pred).replace('Q',''))
                    ent_pred[surface] = pred
        if not evaluate:
            if predict_only or score_only:
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
        pred = pred.fillna('{}').map(json.loads)
        
        pred = pred.map(lambda x: x if not type(x)==float else {} )
        gold = gold.map(lambda x: x if not type(x)==float else {} )
        
        preddf = pd.DataFrame({'gold':gold, 'pred':pred })
        import numpy as np
        preddf = preddf.replace([np.nan], [None])
        scores[predfile.stem] = get_scores(preddf.gold, preddf.pred)
        
    if 'defopt' in sys.modules:
        pd.set_option('display.max_colwidth', None)
        print(pd.DataFrame(scores).T)
    else:
        return pd.DataFrame(scores).T
    
def experiment(
    dawgfile: pathlib.Path,
    filtered_counts_file: pathlib.Path,
    infile: pathlib.Path,
    root: pathlib.Path = pathlib.Path('.'),
    *,
    vectorizers: typing.List[pathlib.Path] = (),
    stem: str = None, # TODO
    countfile: pathlib.Path = None,
):
    """
    Run experiment
    """
    if not countfile:
        countfile = (root / f'{filtered_counts_file.stem.rsplit(".", 1)[0]}.json')
    logging.info(f'Using count file {countfile}')
    
    predfiles = []
    models_vectorizers = [(None,None)]
    filt = filtered_counts_file.stem
    ext = 'vw'
    
    # Create feature hashing model
    dim = 2**18
    for bal in [False]:#, True]:
        b = '.bal' if bal else ''
        hash_filestem = f'{filt}.hash{dim}{b}'
        hashmodelfile = pathlib.Path(f'{hash_filestem}.{ext}')
        if not list(glob.glob(hash_filestem + '*.dat')):
            logging.info(f'No vectors {hash_filestem}*.dat, creating...')
            hash_file = vectorize(paragraphlinks_dir, filtered_counts_file, dim=dim, balanced=bal)
        else:
            hash_file = list(glob.glob(hash_filestem + '*.dat'))[0]
        if not hashmodelfile.exists():
            logging.info(f'No model {hashmodelfile}, creating...')
            train(filtered_counts_file, hash_file)
        models_vectorizers.append( (hashmodelfile, None) )

    # Create custom vectorizer models
    for vec in vectorizers:
        for bal in [False]:#, True]:
            b = '.bal' if bal else ''
            vec_filestem = f'{filt}.{vec.stem}{b}'
            vecmodelfile = pathlib.Path(f'{vec_filestem}.{ext}')
            if not list(glob.glob(vec_filestem + '*.dat')):
                logging.info(f'No vectors {vec_filestem}*.dat, creating...')
                vec_file = vectorize(paragraphlinks_dir, filtered_counts_file, vectorizer=vec, balanced=bal)
            else:
                vec_file = list(glob.glob(vec_filestem + '*.dat'))[0]
            if not vecmodelfile.exists():
                logging.info(f'No model {vecmodelfile}, creating...')
                train(filtered_counts_file, vec_file)
            models_vectorizers.append( (vecmodelfile, vec) )

    for count_arg in (None, countfile):
        for modelfile, vectorizer in models_vectorizers:
            outfile = 'pred-mewsli-'
            if modelfile:
                outfile += modelfile.stem
            else:
                outfile += 'base'
            outfile += ('-' + count_arg.stem) if count_arg else ''
            outfile += '.tsv'
            predfiles.append( pathlib.Path(outfile) )
            logging.info(f'Writing to {outfile}')
            with open(outfile, 'w') as f, redirect_stdout(f):
                run(
                    dawgfile,
                    filtered_counts_file,
                    modelfile,
                    infile,
                    vectorizer=vectorizer,
                    lang=stem,
                    countfile=count_arg,
                    predict_only=True,
                )
    
    evaluate(
        infile,
        *sorted(predfiles),
    )
