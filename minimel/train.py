"""
Train
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging

from vowpalwabbit import pyvw


def train(
    surface_score_file: pathlib.Path,
    vec_file: pathlib.Path,
    *,
    bits: int = 32,
    ldf: bool = False,
):
    """
    Train Logistic Regression models

    Writes `{vec_file}.vw`
    
    Args:
        surface_score_file: JSON file of surfaceform-entity scores
        vec_file: Training data in Vowpal Wabbit format
    """

    vec_file = pathlib.Path(vec_file)
    b = f'.{bits}b'
    l = '.ldf' if ldf else ''
    fname = str(vec_file.parent / f'{vec_file.stem}{b}{l}.vw')
    
    params = dict(
        data=str(vec_file),
        final_regressor=fname,
        bit_precision=bits,
        invert_hash=fname+'.inverted', # https://stackoverflow.com/a/24660302
        # readable_model=fname+'.readable',
        # passes=10,
        # cache=True
    )
    
    if ldf:
        params['csoaa_ldf'] = 'multiline'
        params['quadratic'] = 'ls'
    else:
        ents = set( int(label.split(':')[0])
            for line in vec_file.open() 
            for label in line.split('|', 1)[0].split()
            if label
        )
        params['csoaa'] = max(ents)+1
    
    pyvw.Workspace(**params)
    
    logging.info(f"Wrote to {fname}")
    return fname
