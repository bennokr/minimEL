"""
Filter anchor counts
"""
import math, collections, itertools
import json, sqlite3, tqdm, re, html
import sys, pathlib, argparse, logging

from .normalize import normalize


def get_titles(indexdbfile, ents=None, surfaceforms=None, language=None):
    """Get Wikipedia article titles"""
    db = sqlite3.connect(indexdbfile)
    (total,) = next(db.execute("select count(*) from mapping"))
    rows = db.execute("select * from mapping")
    if logging.root.level < 30:
        (total,) = next(db.execute("select count(*) from mapping"))
        rows = tqdm.tqdm(rows, total=total, desc="Loading labels...")
    title_ids, id_titles = {}, {}
    for pageid, pagetitle, wikidata_id in rows:
        if pagetitle and wikidata_id:
            for title in normalize(pagetitle, language=language):
                wi = int(wikidata_id[1:])
                if (not ents) or (wi in ents) or (title in surfaceforms):
                    title_ids.setdefault(title, set()).add(wi)
                    id_titles.setdefault(wi, set()).add(title)
    return title_ids, id_titles


## Count filterint (not used)
def steps(count):
    logs = [math.log(c) + 1 for e, c in count.most_common()]
    total = sum(logs)
    return [0] + [(logs[i] - l) / logs[i] for i, l in enumerate(logs[1:])]


def filter_steps(ent_count, cutoff=0.7):
    new_ent_count = collections.Counter()
    for s, (e, c) in zip(steps(ent_count), count.most_common()):
        if s > cutoff:
            break
        new_ent_count[e] = c
    return new_ent_count


def filter_counts_cutoff(s_e_count, cutoff=0.7):
    for a, ec in s_e_count.items():
        s_e_count[a] = filter_steps(ec, cutoff=cutoff)
    return s_e_count


## Surfaceform Filtering
def entropy(ent_count):
    t = sum(ent_count.values())
    return -sum((v / t) * math.log(v / t) for v in ent_count.values())


def countratio(ent_count):
    return len(ent_count) / sum(ent_count.values())


import re


def tokens(s, N=3):
    s = s.lower().rsplit(" (")[0].rsplit(" ,")[0]
    return set(
        w[i : i + N]
        for w in re.split("\W", s.lower())
        if w.strip()
        for i in range(len(w) - N + 1)
    )


def tokenscore(surfaceform, count, id_titles):
    """Calculate the average mean token overlap between
    the surface form & the titles of candidate labels
    (asymmetric jaccard index)
    """
    stok = tokens(surfaceform)
    if not stok:
        return 0
    leftjacc = lambda a, b: len(a & b) / len(a)
    alltoks = lambda c: set(t for l in id_titles.get(c, "") for t in tokens(l))
    return sum(leftjacc(stok, alltoks(c)) for c in count) / len(count)


def clean(
    indexdbfile: pathlib.Path,
    disambigfile: pathlib.Path,
    countfile: pathlib.Path,
    *,
    lang: str = None,
    freqnorm: bool = False,
    badentfile: pathlib.Path = None,
    mincount: int = 2,
    tokenscore_threshold: float = 0.1,
    entropy_threshold: float = 1.0,
    countratio_threshold: float = 0.5,
    shadowed_top: float = None,
):
    """
    Filter anchor counts (given their candidate entity counts).

    First, only keep candidate entities that either have minimal counts or are
    linked from disambiguation pages.
    If the tokenscore is low, then surfaceforms with high entropy or countratio
    (len / sum) are removed.

    Writes `filtered_counts.{n_good_counts}.json`

    Args:
        indexdbfile: Wikimapper index sqlite3 database
        disambigfile: Disambiguation JSON file
        countfile: Hyperlink anchor count {word: {Q_ent: count}} JSON file

    Keyword Arguments:
        lang: Stemming language ISO 639-1 (2-letter) code
        freqnorm: Normalize counts by total entity frequency
        badentfile: Entity IDs to ignore, one per line
        mincount: Minimal candidate entity count
        tokenscore_threshold: Threshold for mean asymmentric Jaccard index
            between surface form and candidate entity labels
        entropy_threshold: Entropy threshold (high entropy = flat dist)
        countratio_threshold: Count-ratio (len / sum) threshold
        shadowed_top: Only train models for a % surfaceforms with highest counts
            of candidate entities shadowed by the top candidate
    """
    surface_ent_counts = json.load(open(countfile))
    total_ent_count = collections.Counter()
    ss = surface_ent_counts.items()
    if logging.root.level < 30:
        ss = tqdm.tqdm(ss, desc="Counting entities...")
    for s, ec in ss:
        ec = {int(e[1:]):c for e, c in ec.items()}
        for e,c in ec.items():
            total_ent_count[e] += c
        surface_ent_counts[s] = collections.Counter(ec)
    
    disambig_surfaceforms = set()
    for s, es in json.load(open(disambigfile)).items():
        for n in normalize(s, language=lang):
            disambig_surfaceforms.add(n)
            for e in es:
                surface_ent_counts.setdefault(n, collections.Counter())[e] += 1
    
    # Filter out bad entities
    ents = set(total_ent_count)
    badents = set(int(q.replace("Q", "")) for q in open(badentfile).readlines())
    logging.info(f"Removing {len(badents & ents)} bad entities")
    ents -= badents
    for a, ec in surface_ent_counts.items():
        norm = {}
        if freqnorm:
            norm = {e:total_ent_count.get(e,1) for e in ec}
            maxnorm = max(norm.values())
            norm = {e:c / maxnorm for e,c in norm.items()}
        surface_ent_counts[a] = {
            e: int(c * norm.get(e, 1))+1
            for e, c in ec.items() 
            if e in ents and c >= mincount
        }
    surface_ent_counts = {s:ec for s,ec in surface_ent_counts.items() if ec}

    title_ids, id_titles = get_titles(
        str(indexdbfile), ents, set(surface_ent_counts), language=lang)

    # Collect bad surface forms
    high_entropy, high_countratio, no_tokenmatch = set(), set(), set()
    ss = surface_ent_counts.items()
    if logging.root.level < 30:
        ss = tqdm.tqdm(ss, desc="Filtering surfaceforms...")
    for s, ent_count in ss:
        # Disambiguation surfaceforms are always good
        if s in disambig_surfaceforms:
            continue
        tscore = tokenscore(s, ent_count, id_titles)
        if tscore < tokenscore_threshold:
            # If the tokenscore is low,
            # then surfaceforms with high entropy or countratio are bad
            if entropy(ent_count) > entropy_threshold:
                high_entropy.add(s)
            elif countratio(ent_count) > countratio_threshold:
                high_countratio.add(s)
        elif (tscore == 0) and (s not in title_ids):
            # this might happen for unknown abbreviations or weird names
            # possible extension: track possible abbreviations
            no_tokenmatch.add(s)

    bad_surfaceforms = high_entropy | high_countratio | no_tokenmatch
    logging.info(f"Filtering out {len(bad_surfaceforms)} bad surfaceforms")

    good_counts = {
        s: dict(sorted(ec.items(), key=lambda x: -x[1]))
        for s, ec in surface_ent_counts.items()
        if s not in bad_surfaceforms and ec
    }
    logging.info(f"Keeping {len(good_counts)} good surfaceforms")
    
    if shadowed_top:
        top_counts, shadow_counts = {}, []
        for s, ec in good_counts.items():
            if len(ec) > 1:
                top, *shadow = ec.items()
                top_counts[s] = top
                for e,c in shadow:
                    shadow_counts.append( (c, s, e) )
        good_counts = {}
        shadowed_topn = int(shadowed_top * len(shadow_counts))
        for c,s,e in sorted(shadow_counts)[::-1][:shadowed_topn]:
            te, tc = top_counts[s]
            good_counts.setdefault(s, {})[te] = tc
            good_counts[s][e] = c

    print(json.dumps(good_counts))
