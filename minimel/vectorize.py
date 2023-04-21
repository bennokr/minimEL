"""
Vectorize paragraph text dataset
"""
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pathlib, argparse, logging
import regex as re
import html
import pickle
import json

import dawg
import tqdm
import pandas as pd
import numpy as np

from .normalize import normalize

token_pattern = re.compile(r"(?u)\b\w+\b")
    
def vw_tok(text):
    return [t for t in token_pattern.findall(text.lower()) if ('|' not in t) and (':' not in t)]

def vw(lines, anchor_json: pathlib.Path, ent_feats_csv=None, balanced=False, language=False, usenil=False):
    """Create VW-formatted training data
    
    Args:
        lines: iterable of (pageid, {name: entityid} json, text) tsv lines
        anchor_json: path to json file of {name: {entityid: weight}}
        ent_feats_csv: path to csv of (entityid,feat1 feat2 feat3 ...)
        
    """
    surface_weights = json.load(anchor_json.open())
    surface_weights = {
        m:{int(e.replace('Q','')):c for e,c in ec.items()} 
        for m, ec in surface_weights.items()
    }
    
    if usenil:
        # Make surfaceform lookup trie
        surface_trie = dawg.CompletionDAWG(surface_weights)
    
    # Load entity features
    ent_feats = None
    if ent_feats_csv:
        ent_feats = pd.read_csv(ent_feats_csv, header=None, index_col=0, na_values='')[1]
        ent_feats = ent_feats.fillna('')
        logging.info(f'Loaded {len(ent_feats)} entity features')
    
    def vw_label_lines(weights, e, norm, ent_feats=None):
        if len(weights) > 1:
            entlabels = [(int(e),0)] # positive
            for o,c in weights.items(): # negatives
                if o != e:
                    w = int(np.log1p(c)) if balanced else 1
                    entlabels += [(int(o),w)]
            ns = '_'.join(vw_tok(norm))
            for l,w in entlabels:
                efeats = str(ent_feats.get(l, '')) if (ent_feats is not None) else ''
                efeats = '|f ' + efeats if efeats else ''
                yield f'{l}:{w} |l {ns}={l} {efeats}'
    
    outlines = []
    for line in lines:
        pgid, mention_ent, text = line.split('\t', 2)
        try:
            mention_ent = json.loads(mention_ent)
        except ValueError:
            print(mention_ent)
            raise
        
        tokens = vw_tok(text) # Tokenize
        if not tokens:
            continue
        labels = []
        for m, e in mention_ent.items():
            for norm in normalize(m, language=language):
                if norm in surface_weights and e in surface_weights[norm]:
                    weights = surface_weights[norm]
                    for label in vw_label_lines(weights, e, norm, ent_feats):
                        labels.append(label)
        
        if usenil:
            # add NIL data from surface_trie
            for normtext in normalize(text, language=language):
                normtoks = vw_tok(normtext)
                for i,tok in enumerate(vw_tok(normtext)):
                    for comp in surface_trie.keys(tok):
                        if comp not in mention_ent:
                            comp_toks = vw_tok(comp)
                            if tokens[i:i+len(comp_toks)] == comp_toks:
                                # NIL match
                                weights = surface_weights[comp]
                                for label in vw_label_lines(weights, -1, comp, ent_feats):
                                    labels.append(label)
        
        if labels:
            outlines += [f'shared |s ' + ' '.join(tokens)] + labels + ['']
    outlines.append('')
    return outlines


class TransLiterator:
    def __init__(self, lang):
        import requests
        url = f'https://raw.githubusercontent.com/snowballstem/snowball/master/algorithms/{lang}.sbl'
        resp = requests.get(url)
        if resp.ok:
            defs = [l[9:].split(None, 1) for l in resp.text.splitlines() if l.startswith('stringdef')]            
            self.charmap = {f'\\u{code[4:-2]}'.encode().decode('unicode_escape'):name for name, code in defs}
    
    def code(self, text):
        for a,b in self.charmap.items():
            text = text.replace(a, b)
        return text
                        
def hashvec(paragraphs, dim=None, lang=None, tokenizer=None):
    from sklearn.feature_extraction.text import HashingVectorizer

    if lang:
        from icu_tokenizer import Tokenizer

        tokenizer = Tokenizer(lang=lang).tokenize
    vec = HashingVectorizer(
        n_features=(dim or 2**18),
        tokenizer=tokenizer,
    )
    return vec.fit_transform(paragraphs)


def transform(paragraphs, vectorizer):
    vec = pickle.load(open(vectorizer, "rb"))
    return vec.transform(paragraphs)


def embed(paragraphs, embeddingsfile, dim=None):
    warnings.simplefilter(action="ignore", category=Warning)
    import fasttext

    m = fasttext.load_model(str(embeddingsfile))
    return np.vstack([m.get_sentence_vector(p)[:dim] for p in paragraphs])


def vectorize(
    paragraphlinks: pathlib.Path,
    anchor_json: pathlib.Path,
    *,
    vectorizer: pathlib.Path = None,
    ent_feats_csv: pathlib.Path = None,
    lang: str = None,
    balanced: bool = False,
):
    """
    Vectorize paragraph text dataset into Vowpal Wabbit format

    Args:
        paragraphlinks: Paragraph links directory
        anchor_json: Anchor count json file
        vectorizer: Scikit-learn vectorizer .pickle or Fasttext .bin word
            embeddings. If unset, use HashingVectorizer.
        ent_feats_csv: CSV of (ent_id,space separated feat list) entity features
        lang: ICU tokenizer language
        balanced: balanced training
    """
                
    if vectorizer:
        name = anchor_json.stem + "." + vectorizer.stem
    else:
        name = anchor_json.stem
    
    b = '.bal' if balanced else ''
    f = f'.{ent_feats_csv.stem}' if ent_feats_csv else ''
    fname = (anchor_json.parent / f'{name}{b}{f}.parts')
    logging.info(f"Writing to {fname}")
    
    import dask.bag as db
    from .scale import progress, get_client
    
    if lang:
        logging.info(f"Snowball stemming for language: {lang}")
    
    
    
    with get_client():
        urlpath = str(paragraphlinks)
        if paragraphlinks.is_dir():
            urlpath += "/*" 
            
        bag = db.read_text(urlpath, files_per_partition=3)
        data = (
            bag.map_partitions(
                vw, anchor_json, 
                ent_feats_csv=ent_feats_csv,
                balanced=balanced,
                language=lang)
            .to_textfiles(str(fname))
        )
        # with open(fname, 'w') as fw:
        #     for line in data:
        #         print(line, file=fw)
        return fname
