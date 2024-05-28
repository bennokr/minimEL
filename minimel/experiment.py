import typing
import pathlib
import logging
import itertools
import time
import datetime

from .get_disambig import get_disambig
from .get_paragraphs import get_paragraphs
from .count import count
from .clean import clean
from .vectorize import vectorize
from .train import train
from .run import run


class log_time(object):
    def __init__(self, fname):
        self.fname = fname

    def __enter__(self):
        self.fw = open(self.fname, "w")
        self.start = time.time()
        print(f"Started at {datetime.datetime.now().isoformat()}", file=self.fw)
        return self.fw

    def __exit__(self, type, value, traceback):
        print(f"Ended at {datetime.datetime.now().isoformat()}", file=self.fw)
        print(f"Elapsed: {time.time() - self.start}", file=self.fw)
        self.fw.close()


def find(directory, glob):
    try:
        found = next(directory.glob(glob)).absolute()
        logging.info(f"Using {found}")
        return found
    except StopIteration:
        raise Exception(f"File {glob} not found in {directory}")


def sweep(**kw):
    logging.info(f"Sweeping parameters {kw}")
    return (dict(zip(kw, x)) for x in itertools.product(*kw.values()))


def make_dir_params(name, **params):
    params = {k: v.name if type(v) == pathlib.Path else v for k, v in params.items()}
    param_strs = [f"{k.replace('_','-')}_{v}" for k, v in params.items() if v]
    return "__".join([name, *param_strs])


def get_dir_params(dirname: pathlib.Path):
    for param in dirname.name.split("__")[1:]:
        if "_" in param:
            k, v = param.split("_", 1)
            k = k.replace("-", "_")
            try:
                v = int(v)
            except:
                try:
                    v = float(v)
                except:
                    pass
            yield k, v
        elif param.startswith("no-"):
            yield param.replace("no-", ""), False
        else:
            yield param, True


def experiment(
    root: pathlib.Path = pathlib.Path("."),
    *,
    outdir: pathlib.Path = None,
    nparts: int = 100,
    head: int = None,
    split: typing.List[int] = (None,),
    fold: typing.List[int] = (None,),
    # Count
    stem: typing.List[str] = ("",),
    min_count: typing.List[int] = (2,),
    # Clean
    freqnorm: typing.List[bool] = (False,),
    badentfile: typing.List[pathlib.Path] = ("",),
    tokenscore_threshold: typing.List[float] = (0.1,),
    entropy_threshold: typing.List[float] = (1.0,),
    countratio_threshold: typing.List[float] = (0.5,),
    quantile_top_shadowed: typing.List[float] = (0,),
    cluster_threshold: typing.List[float] = (None,),
    # Vectorize
    vectorizer: typing.List[pathlib.Path] = ("",),
    ent_feats_csv: typing.List[pathlib.Path] = ("",),
    balanced: typing.List[bool] = (False,),
    usenil: typing.List[bool] = (False,),
    # Train
    bits: typing.List[int] = (20,),
    # Run
    runfile: typing.List[pathlib.Path] = ("",),
    use_fallback: typing.List[bool] = (True,),
    also_baseline: bool = True,
    evaluate: bool = False
):
    """
    Run all steps to train and evaluate EL models over a parameter sweep.

    The root directory must contain the following files:

    - `index_*.dawg`: DAWG trie mapping of article names -> numeric IDs
    - `*-disambig.txt`: See `disambig_ent_file` in :obj:`~get_disambig.get_disambig`

    Args:
        root: Root directory

    Keyword Arguments:
        outdir: Write outputs to this directory
        nparts: Number of parts to chunk wikidump into
        head: Use only N first lines from each partition
        split: Split the data into several parts
        fold: Ignore this fold of the split data in training, use in evaluation
        stem: Stemming language ISO 639-1 (2-letter) code (use X for no stemming)
        min_count: Minimal (anchor-text, target) occurrence
        freqnorm: Normalize counts by total entity frequency (1/0)
        badentfile: File of entity IDs to ignore, one per line (default: `*-disambig.txt`)
        tokenscore_threshold: Threshold for mean asymmentric Jaccard index
            between name and candidate entity labels
        entropy_threshold: Entropy threshold (high entropy = flat dist)
        countratio_threshold: Count-ratio (len / sum) threshold
        quantile_top_shadowed: Only train models for a % names with highest counts
            of candidate entities shadowed by the top candidate
        cluster_threshold: Cluster names based on their meanings
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use tokens directly.
        ent_feats_csv: CSV of (ent_id,space separated feat list) entity features
        balanced: Use balanced training
        usenil: Use NIL option for training unlinked mentions
        bits: Number of bits of the Vowpal Wabbit feature hash function
        runfile: TSV rows of (ID, {name -> ID}, text) or ({name -> ID}, text)
        use_fallback: Use raw counts as fallback
        also_baseline: Also run a baseline model without model predictions
        evaluate: Write evaluation scores to file
    """
    results = []
    root = root.absolute()
    outdir = outdir or root
    logging.info(f"Running experiments for {root}, outputs in {outdir}")

    dawgfile = find(root, "index_*.dawg")

    # Get disambiguations
    disambig_ent_file = find(root, "*-disambig.txt")
    if not any(root.glob("disambig.json")):
        logging.info(f"Getting disambiguations...")
        wikidump = find(root, "*-pages-articles.xml")
        get_disambig(wikidump, dawgfile, disambig_ent_file, nparts=nparts)
    disambigfile = find(root, "disambig.json")

    # Get paragraph links
    if not any(root.glob("*-paragraph-links")):
        wikidump = find(root, "*-pages-articles.xml")
        logging.info(f"Getting paragraph links...")
        get_paragraphs(wikidump, dawgfile)
    paragraphlinks = find(root, "*-paragraph-links")

    # Count mentions
    curdir = outdir
    for count_params in sweep(head=[head], stem=stem, min_count=min_count, split=split, fold=fold):
        newdir = curdir / make_dir_params("count", **count_params)
        if not any(newdir.glob("count*.json")):
            newdir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Counting in {newdir}...")
            count(paragraphlinks, outfile=newdir, **count_params)

    for countfile in curdir.rglob("count*/count*.json"):
        curdir = countfile.parent
        count_params = dict(get_dir_params(curdir))

        # Clean mention counts
        for clean_params in sweep(
            stem=[count_params.get("stem")],
            min_count=[count_params.get("min_count", 2)],
            freqnorm=freqnorm,
            badentfile=badentfile,
            tokenscore_threshold=tokenscore_threshold,
            entropy_threshold=entropy_threshold,
            countratio_threshold=countratio_threshold,
            quantile_top_shadowed=quantile_top_shadowed,
            cluster_threshold=cluster_threshold,
        ):
            newdir = curdir / make_dir_params("clean", **clean_params)
            newdir.mkdir(parents=True, exist_ok=True)
            if not any(newdir.glob("clean*.json")):
                logging.info(f"Cleaning in {newdir}...")
                indexdbfile = find(root, "index_*.db")
                if not clean_params["badentfile"]:
                    clean_params["badentfile"] = disambig_ent_file
                clean(
                    indexdbfile, disambigfile, countfile, outfile=newdir, **clean_params
                )

        for cleanfile in curdir.rglob("clean*/clean*.json"):
            curdir = cleanfile.parent
            clean_params = dict(get_dir_params(curdir))

            # Vectorize
            for vec_params in sweep(
                head=[head],
                stem=[count_params.get("stem")],
                vectorizer=vectorizer,
                ent_feats_csv=ent_feats_csv,
                balanced=balanced,
                usenil=usenil,
                split=[count_params.get("split")], 
                fold=[count_params.get("fold")],
            ):
                newdir = curdir / make_dir_params("vec", **vec_params)
                newdir.mkdir(parents=True, exist_ok=True)
                if not any(newdir.glob("vec*.dat")):
                    vectorize(
                        paragraphlinks,
                        cleanfile,
                        outfile=newdir,
                        **vec_params,
                    )

            for vecfile in curdir.rglob("vec*/vec*.dat"):
                curdir = vecfile.parent
                vec_params = dict(get_dir_params(curdir))

                # Train
                for train_params in sweep(
                    bits=bits,
                ):
                    newdir = curdir / make_dir_params("train", **vec_params)
                    newdir.mkdir(parents=True, exist_ok=True)
                    if not any(newdir.glob("model*.vw")):
                        with log_time(newdir / "time.log"):
                            train(vecfile, outfile=newdir, **train_params)

                for trainfile in curdir.rglob("train*/model*.vw"):
                    curdir = trainfile.parent
                    train_params = dict(get_dir_params(curdir))

                    # Run
                    if any(runfile) or split:
                        if split:
                            # if performing cross-validation, use training data
                            runfile = [paragraphlinks]
                        
                        for run_params in sweep(
                            runfile=runfile,
                            use_fallback=use_fallback,
                            split=[count_params.get("split")], 
                            fold=[count_params.get("fold")],
                        ):
                            newdir = curdir / make_dir_params("run", **vec_params)
                            newdir.mkdir(parents=True, exist_ok=True)

                            run_params['fallback'] = (
                                countfile if run_params.pop("use_fallback") else None
                            )
                            sweep_runfile = run_params.pop("runfile")
                            params = {
                                'count': count_params,
                                'clean': clean_params,
                                'vec': vec_params,
                                'train': train_params,
                                'run': run_params,
                            }
                            if also_baseline:
                                logging.info("Running baseline...")
                                with log_time(newdir / "baseline-time.log"):
                                    e = run(
                                        dawgfile,
                                        cleanfile,
                                        None, # No model
                                        sweep_runfile,
                                        outfile=newdir / "run___baseline.tsv",
                                        evaluate=evaluate,
                                        evalfile=newdir / "run___baseline_eval.csv",
                                        **run_params,
                                    )
                                    results.append(({'model':'baseline', **params},e))
                            logging.info("Running model...")
                            with log_time(newdir / "model-time.log"):
                                e = run(
                                    dawgfile,
                                    cleanfile,
                                    trainfile,
                                    sweep_runfile,
                                    outfile=newdir / "run___model.tsv",
                                    evaluate=evaluate,
                                    evalfile=newdir / "run___model_eval.csv",
                                    **run_params,
                                )
                                results.append(({'model':'model', **params},e))
    
    if evaluate:
        import pandas as pd

        params, evals = zip(*results)
        evals = pd.concat(evals, axis=1).T
        evals.columns = evals.columns.map('.'.join)
        df = pd.concat([pd.json_normalize(params), evals], axis=1)
        df.to_csv('evaluation.csv')
