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
    outfile: pathlib.Path = None,
    bits: int = 20,
):
    """
    Train Logistic Regression models

    Writes

    Args:
        vec_file: Training data in Vowpal Wabbit format

    Keyword Arguments:
        outfile: Output file or directory (default: `model.b{bits}.vw`)
        bits: Number of bits of the Vowpal Wabbit feature hash function
    """

    vec_file = pathlib.Path(vec_file)
    b = f".{bits}b"
    fname = f"model{b}.vw"
    if not outfile:
        outfile = vec_file.parent / fname
    if outfile.is_dir():
        outfile = outfile / fname
    outfile.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing to {outfile}")

    params = dict(
        data=str(vec_file),
        final_regressor=str(outfile),
        bit_precision=bits,
        # invert_hash=fname+'.inverted', # https://stackoverflow.com/a/24660302
        # readable_model=fname+'.readable',
        csoaa_ldf="mc",
        loss_function="logistic",
        quadratic=["ls", "sf"],
        probabilities=True,
        passes=10,
        cache=True,
    )

    pyvw.Workspace(**params)

    logging.info(f"Wrote to {fname}")
    return fname
