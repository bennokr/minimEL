"""
Train
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging

from vowpalwabbit import pyvw


def train(
    vec_file: pathlib.Path,
    *,
    bits: int = 20,
):
    """
    Train Logistic Regression models

    Writes `{vec_file}.vw`
    
    Args:
        vec_file: Training data in Vowpal Wabbit format
    """

    vec_file = pathlib.Path(vec_file)
    b = f'.{bits}b'
    fname = str(vec_file.parent / f'{vec_file.stem}{b}.vw')
    
    params = dict(
        data=str(vec_file),
        final_regressor=fname,
        bit_precision=bits,
        # invert_hash=fname+'.inverted', # https://stackoverflow.com/a/24660302
        # readable_model=fname+'.readable',
        
        csoaa_ldf = 'mc',
        loss_function = 'logistic',
        quadratic = ['ls', 'sf'],
        probabilities = True,
        
        passes=10,
        cache=True,
    )
    
    pyvw.Workspace(**params)
    
    logging.info(f"Wrote to {fname}")
    return fname
