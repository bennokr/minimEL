"""
Extract entity features from parquet triples
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import pathlib, logging
import json, collections
import tqdm

def ent_feats(
    spo_parquet: pathlib.Path,
    anchor_json: pathlib.Path,
    *,
    part: float = 1,
):
    """
    Extract entity features from parquet triples
    
    Args:
        spo_parquet: Parquet triple file
        anchor_json: Anchor counts
        part: Filter part of features based on count
            <1: Quantile of feature count
            >1: Minimum feature count
    """
    logging.info(f'Loading triples...')
    spo = pd.read_parquet(spo_parquet).set_index('s')
    logging.info(f'Loaded {len(spo)} triples')
    
    # Load anchor counts
    anchor_dict = json.load(open(anchor_json))
    anchors = [(n,int(k),v) for n,c in anchor_dict.items() for k,v in c.items()]
    anchors = pd.DataFrame(anchors, columns=['n','s','scount'])
    scount = anchors.groupby('s')['scount'].sum()
    
    top_ents = ', '.join(str(e) for e in scount.sort_values().tail().index)
    logging.info(f'Top anchor entities: {top_ents}')
    
    # Take subset of relevant triples
    spo = spo.join(scount, how='inner')
    logging.info(f'Got {len(spo)} relevant triples')
    
    # Get feature counts
    pocount = spo.groupby(['p', 'o'])['scount'].count().rename('pocount')
    pocount = pocount.sort_values(ascending=False)
    if part > 1:
        part = int(part)
    pocount = pocount[pocount >= (pocount.quantile(1-part) if part<1 else part)]
    
    # Collect features per entity
    sfeats = {}
    for s in tqdm.tqdm(scount.index):
        feats = spo.loc[s:s].merge(pocount.reset_index())
        f = 'P' + feats['p'].astype('str') + 'Q' + feats['o'].astype('str')
        sfeats[s] = ' '.join(set(f))
    
    fname = f'feat-{anchor_json.stem}.p{part}.csv'
    pd.Series(sfeats).to_csv(fname, header=None)
    logging.info(f'Wrote to {fname}')