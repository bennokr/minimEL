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
from vowpalwabbit import pyvw


def undersample(X, y, n):
    if n < X.shape[0]:
        
        # Always re-weight subsampling
        weights = pd.Series(y).replace(1 / pd.Series(y).value_counts())
            
        i = pd.DataFrame(range(X.shape[0])).sample(n=n, weights=weights)
        return X[i.index], np.array(y)[i.index]
    else:
        return X, y


def train(
    surface_score_file: pathlib.Path,
    vec_file: pathlib.Path,
    max_samples: int = None,
    max_features: int = 300,
    *,
    unbalanced: bool = False,
):
    """
    Train Logistic Regression models

    Writes `{vec_file}.vwmodel`
    
    Args:
        surface_score_file: JSON file of surfaceform-entity scores
        vec_file: Training data in Vowpal Wabbit format
        max_samples: Maximum nr of training samples
        max_features: Select features with highest coefficients
    """
    scores = json.load(open(surface_score_file))
    ents = set(int(e.replace('Q','')) for es in scores.values() for e in es)

    vec_file = pathlib.Path(vec_file)
    u = '.unbal' if unbalanced else ''
    fname = str(vec_file.parent / f'{vec_file.stem}{u}.vw')
    pyvw.Workspace(
        data=str(vec_file),
        final_regressor=fname,
        csoaa=max(ents),
        cache=True
    )
    return fname
