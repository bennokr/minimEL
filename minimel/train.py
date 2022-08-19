"""
Train
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp


def undersample(X, y, n, class_weight=None):
    if n < X.shape[0]:
        weights = None
        if class_weight == 'balanced':
            weights = pd.Series(y).replace(1 / pd.Series(y).value_counts())
        else:
            # make sure all classes are in the subsample!
            pass
        i = pd.DataFrame(range(X.shape[0])).sample(n=n, weights=weights)
        return X[i.index], np.array(y)[i.index]
    else:
        return X, y


def get_coef_df(clf, rounding=4, max_features=300):
    coef = clf.coef_.copy()
    if sp.issparse(coef):
        coef.data = np.round(coef.data, rounding)
        coef = [
            sorted(
                [[int(a), b] for a, b in zip(coef[i].indices, coef[i].data)],
                key=lambda x: -abs(x[1]),
            )[:max_features]
            for i in range(coef.shape[0])
        ]
    else:
        coef = np.round(coef, rounding)
        coef = map(list, coef)

    return pd.concat(
        [
            pd.Series(clf.classes_, name="class"),
            pd.Series(clf.intercept_, name="intercept"),
            pd.Series(coef, name="coef"),
        ],
        axis=1,
    )


def train_models(surface_entscore, clf, vec_filestem, max_samples=None, max_features=300, class_weight=None):

    if pathlib.Path(str(vec_filestem) + ".npy").exists():
        arr = np.load(str(vec_filestem) + ".npy", mmap_mode="r")
    elif pathlib.Path(str(vec_filestem) + ".npz").exists():
        import scipy.sparse

        arr = scipy.sparse.load_npz(str(vec_filestem) + ".npz")
    else:
        raise FileNotFoundError(vec_filestem)

    index = pd.read_parquet(str(vec_filestem) + ".index.parquet")
    by_surface, by_ent = index.by_surface, index.by_ent

    coefs = []
    for surfnum, surface, ent_score in surface_entscore:
        # Get dataset
        y = by_ent[by_surface == surfnum]
        y[~y.isin(ent_score)] = -1
        
        if len(set(y)) > 1:
            X, y = arr[y.index], np.array(y)
            if max_samples is not None:
                X, y = undersample(X, y, max_samples, class_weight=class_weight)
            
            clf.fit(X, y)
            
            if (clf.coef_ == 0).mean() > 0.5:
                clf.sparsify()

            df = get_coef_df(clf, max_features=max_features)
            support = pd.Series(y).value_counts()
            df.insert(1, "support", [support[e] for e in clf.classes_])
            df.insert(0, "surfaceform", surface)
            coefs.append(df)
    return [pd.concat(coefs)] if coefs else []


def train(
    surface_score_file: pathlib.Path,
    vec_filestem: pathlib.Path,
    max_samples: int = None,
    max_features: int = 300,
    *,
    unbalanced: bool = False,
):
    """
    Train Logistic Regression models

    Writes `{vec_file}.logreg.parquet`

    Args:
        surface_score_file: JSON file of surfaceform-entity scores
        vec_filestem: Training data vector files (.npy/.npz and .index.parquet)
        max_samples: Maximum nr of training samples
        max_features: Select features with highest coefficients
    """
    import dask.bag as db
    from .scale import progress, get_client

    with get_client() as client:
        
        logging.info(f'Using surface scores {surface_score_file}')
        anchor_scores = json.load(open(surface_score_file))
        scores = [
            (i, surface, {int(e.replace("Q", "")): s for e, s in es.items()})
            for i, (surface, es) in enumerate(anchor_scores.items())
        ]
        logging.info(f"Fitting {len(scores)} models")

        from sklearn.linear_model import LogisticRegression
        
        class_weight = None if unbalanced else 'balanced'
        clf = LogisticRegression(solver="liblinear", C=1, class_weight=class_weight)

        bag = db.from_sequence(scores)
        client.persist(bag)

        coefs = bag.map_partitions(
            train_models,
            clf,
            vec_filestem,
            max_samples=max_samples,
            max_features=max_features,
            class_weight=class_weight,
        )
        
        fname = str(vec_filestem)
        if max_samples:
            fname += ".max" + str(max_samples)
        if unbalanced:
            fname += ".unbal"
        fname += ".logreg.parquet"
        
        logging.info(f"Training {fname} models")
        progress(coefs.persist())
        
        df = pd.concat(coefs.compute()).set_index(["surfaceform", "class"])
        logging.info(f"Writing models to {fname}")
        df.to_parquet(fname)
        return fname
