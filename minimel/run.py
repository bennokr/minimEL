import typing
import pathlib
import pickle
import json
import sys
import glob
import logging
from contextlib import redirect_stdout

import tqdm
import pandas as pd
import numpy as np
from vowpalwabbit import pyvw

try:
    import dawg
except ImportError:
    import dawg_python as dawg

from .normalize import normalize
from .vectorize import hashvec, transform, embed, vectorize, vw_tok


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


class MiniNED:
    def __init__(
        self,
        dawgfile: pathlib.Path,
        candidatefile: pathlib.Path = None,
        modelfile: pathlib.Path = None,
        vectorizer: pathlib.Path = None,
        ent_feats_csv: pathlib.Path = None,
        lang: str = None,
        fallback: pathlib.Path = None,
    ):
        """
        Named Entity Disambiguation class

        Args:
            dawgfile: DAWG trie file of Wikipedia > Wikidata count
            candidatefile: Candidate {surfaceform -> [ID]} json
            modelfile: Vowpal Wabbit model

        Keyword Arguments:
            vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
                embeddings. If unset, use HashingVectorizer.
            ent_feats_csv: CSV of (ent_id,space separated feat list) entity features
            fallback: Additional fallback deterministic surfaceform -> ID json
        """

        self.lang = lang

        self.index = dawg.IntDAWG()
        self.index.load(str(dawgfile))

        self.candidates = json.load(candidatefile.open()) if candidatefile else {}
        self.count = json.load(fallback.open()) if fallback else {}

        self.ent_feats = None
        if ent_feats_csv:
            self.ent_feats = pd.read_csv(
                ent_feats_csv, header=None, index_col=0, na_values=""
            )
            self.ent_feats = ent_feats[1].fillna("")
            logging.info(f"Loaded {len(self.ent_feats)} entity features")

        self.model = None
        if candidatefile and modelfile:
            self.model = pyvw.Workspace(
                initial_regressor=str(modelfile),
                loss_function="logistic",
                csoaa_ldf="mc",
                probabilities=True,
                testonly=True,
                quiet=True,
            )

    def _model_predict(self, text, norm, ents, all_scores=False):
        # TODO: replace with vectorize.vw!!!
        preds = {}
        ents = list(ents)
        toks = "shared |s " + " ".join(vw_tok(text))
        ns = "_".join(vw_tok(norm))
        for i, ent in enumerate(ents):
            efeats = (
                str(self.ent_feats.get(l, "")) if (self.ent_feats is not None) else ""
            )
            efeats = "|f " + efeats if efeats else ""
            cands = [f"{e} |l {ns}={e} {efeats}" for e in ents[i:] + ents[:i]]
            preds[ent] = self.model.predict([toks] + cands)
        if all_scores:
            return preds
        else:
            return max(preds.items(), key=lambda x: x[1])[0]

    def predict(self, text: str, surface: str, upperbound=None, all_scores=False):
        """
        Make NED prediction.

        Args:
            text: Some text
            surface: An entity name in `text`

        Keyword Arguments:
            all_scores: Output all candidate scores
            upperbound: Create upper bound on performance

        Returns:
            Wikidata ID
        """
        pred = None
        if upperbound:
            gold = str(upperbound)
            for norm in normalize(surface, language=self.lang):
                if (norm in self.count) and (gold in self.count[norm]):
                    pred = gold
            if not pred:
                if str(self.index.get(surface.replace(" ", "_"), -1)) == gold:
                    pred = gold
        else:
            for norm in normalize(surface, language=self.lang):
                ent_cand = self.candidates.get(norm, None)
                if ent_cand and self.model:  # Vowpal Wabbit model
                    pred = self._model_predict(
                        text, norm, ent_cand, all_scores=all_scores
                    )
                elif norm in self.count:  # fallback: most common meaning
                    dist = self.count[norm]
                    pred = max(dist, key=lambda x: dist[x])
                elif surface.replace(" ", "_") in self.index:  # deterministic index
                    pred = self.index[surface.replace(" ", "_")]
        if pred:
            if all_scores:
                if type(pred) != dict:
                    pred = {pred: 1}
                pred = {f"Q{p}": s for p, s in pred.items()}
            else:
                pred = int(str(pred).replace("Q", ""))
        return pred


def run(
    dawgfile: pathlib.Path,
    candidatefile: pathlib.Path = None,
    modelfile: pathlib.Path = None,
    *runfiles: pathlib.Path,
    outfile: pathlib.Path = None,
    vectorizer: pathlib.Path = None,
    ent_feats_csv: pathlib.Path = None,
    lang: str = None,
    fallback: pathlib.Path = None,
    evaluate: bool = False,
    predict_only: bool = True,
    all_scores: bool = False,
    upperbound: bool = False,
):
    """
    Perform entity disambiguation

    Args:
        dawgfile: DAWG trie file of Wikipedia > Wikidata count
        candidatefile: Candidate {surfaceform -> [ID]} json
        modelfile: Vowpal Wabbit model
        runfiles: Input file (- or absent for standard input). TSV rows of
            (ID, {surface -> ID}, text) or ({surface -> ID}, text) or (text)

    Keyword Arguments:
        outfile: Write outputs to file (default: stdout)
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use HashingVectorizer.
        ent_feats_csv: CSV of (ent_id,space separated feat list) entity features
        fallback: Additional fallback deterministic surfaceform -> ID json
        evaluate: Report evaluation scores instead of predictions
        predict_only: Only print predictions, not original text
        all_scores: Output all candidate scores
        upperbound: Create upper bound on performance
    """
    if (not any(runfiles)) or ("-" in runfiles):
        runfiles = (sys.stdin,)
    logging.debug(f"Reading from {runfiles}")
    if (not outfile) or (outfile == "-"):
        outfile = sys.stdout
    else:
        outfile = outfile.open("w")
    logging.debug(f"Writing to {outfile}")

    ned = MiniNED(
        dawgfile,
        candidatefile=candidatefile,
        modelfile=modelfile,
        vectorizer=vectorizer,
        ent_feats_csv=ent_feats_csv,
        lang=lang,
        fallback=fallback,
    )

    ids, ents, texts = (), (), ()
    data = pd.concat([pd.read_csv(i, sep="\t", header=None) for i in runfiles])
    if data.shape[1] == 1:
        texts = data[0]
    elif data.shape[1] == 2:
        ids, texts = data[0], data[1]
    elif data.shape[1] == 3:
        data[1] = data[1].map(json.loads)
        ids, ents, texts = data[0], data[1], data[2]

    preds = []
    it = tqdm.tqdm(texts, "Predicting") if logging.root.level < 30 else texts
    for i, text in enumerate(it):
        ent_pred = {}
        if len(ents):
            for surface in ents[i]:
                gold = ents[i][surface] if upperbound else None
                pred = ned.predict(
                    text, surface, upperbound=gold, all_scores=all_scores
                )
                if pred:
                    ent_pred[surface] = pred
        if not evaluate:
            if predict_only or all_scores:
                if len(ids):
                    print(ids[i], json.dumps(ent_pred), sep="\t", file=outfile)
                else:
                    print(json.dumps(ent_pred), sep="\t", file=outfile)
            else:
                if len(ids):
                    print(ids[i], json.dumps(ent_pred), text, sep="\t", file=outfile)
                else:
                    print(json.dumps(ent_pred), text, sep="\t", file=outfile)
            outfile.flush()

        preds.append(ent_pred)

    if len(ents) and evaluate:
        print(get_scores(ents, preds).T.to_csv(), file=outfile)


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
    for predfile in tqdm.tqdm(predfiles, "Evaluating"):
        data = pd.read_csv(str(predfile), sep="\t", header=None)
        pred = data[0] if data.shape[1] == 1 else data.set_index(0)[1]
        pred = pred.fillna("{}").map(json.loads)

        pred = pred.map(lambda x: x if not type(x) == float else {})
        gold = gold.map(lambda x: x if not type(x) == float else {})

        preddf = pd.DataFrame({"gold": gold, "pred": pred})
        import numpy as np

        preddf = preddf.replace([np.nan], [None])
        scores[predfile] = get_scores(preddf.gold, preddf.pred)

    if "defopt" in sys.modules:
        pd.set_option("display.max_colwidth", None)
        print(pd.DataFrame(scores).T)
    else:
        return pd.DataFrame(scores).T
