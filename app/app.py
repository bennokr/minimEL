import pathlib
from dataclasses import dataclass
import json

from flask import Flask, request, render_template

import minimel
from minimel.mentions import get_matches, setup_matcher

import app_deploy

app = Flask(__name__)
app.add_url_rule('/update_server', methods=['POST'], view_func=app_deploy.update)

@dataclass
class Model:
    name: str
    matcher: None #'AhoCorasick'
    ned: minimel.MiniNED
    eval: pathlib.Path

basedir = pathlib.Path(__file__).parent.parent
app.logger.info(f'Loading models...')
models = {
    "simple": Model(
        name="Simple English",
        matcher=setup_matcher(basedir / "wiki/simplewiki-20211120/count.min2.json"),
        ned=minimel.MiniNED(
            basedir / "wiki/simplewiki-20211120/index_simplewiki-20211120.dawg"
        ),
        eval= basedir / "evaluation/Mewsli-9/en.tsv",
    )
}


@app.route("/")
def index():
    return render_template('index.html', 
        models=models,
    )


def make_links(text, matcher, ned, gold=None):

    # Gazetteer mention detection
    pred_matches = {}
    for start, m in get_matches(matcher, text.lower()):
        m = text[start : start + len(m)]
        pred = ned.predict(text, m)
        if pred:
            pred_matches[(start, start+len(m))] = (m, pred)

    if gold:
        gold_matcher = setup_matcher(None, names=list(gold))
        gold_matches = {}
        for start, m in get_matches(gold_matcher, text):
            gold_matches[(start, start+len(m))] = (m, gold[m])

    # Find connected components of overlapping matches
    positions = sorted(set(pred_matches) | set(gold_matches))
    i, components = 0, {}
    while i < len(positions):
        _, offset = positions[i]
        components[i] = component = set([i])
        # find the next clear position
        while (i:=i+1) < len(positions):
            start, end = positions[i]
            if start < offset:
                component.add(i)
                offset = max(offset, end)
            else:
                break

    offset, out = 0, ''
    for component in components.values():
        preds = set()
        for i in component:
            p = positions[i]
            if p in pred_matches:
                preds.add( (p, pred_matches[p]) )
        
        golds = set()
        for i in component:
            p = positions[i]
            if p in gold_matches:
                golds.add( (p, gold_matches[p]) )
        
        starts, ends = zip(*[positions[i] for i in component])
        start, end = min(starts), max(ends)
        out += text[offset:start]
        if preds == golds:
            # True Positive
            for _, (name, link) in golds:
                out += f'<a class="tp" href="https://www.wikidata.org/wiki/Q{link}">{name}</a>'
        else:
            if preds and golds:
                out += '['
            # Predictions (False Positives)
            sub_offset = start
            for (start, _), (name, link) in sorted(preds):
                out += text[sub_offset:start]
                out += f'<a class="fp" href="https://www.wikidata.org/wiki/Q{link}">{name}</a>'
                sub_offset = start + len(name)
            if preds and golds:
                out += text[sub_offset:end] + ' / '
            # Gold labels (False Negatives)
            sub_offset = start
            for (start, _), (name, link) in sorted(golds):
                out += text[sub_offset:start]
                out += f'<a class="fn" href="https://www.wikidata.org/wiki/Q{link}">{name}</a>'
                sub_offset = start + len(name)
            if preds and golds:
                out += text[sub_offset:end] + ']'
        offset = end
    out += text[offset:]
    return out

@app.route("/random")
def random():
    from random import choice
    lang = request.args.get("lang", None)
    model = models[lang]
    lines =  open(model.eval).readlines()
    _, gold, text = choice(lines).split('\t')
    return json.dumps({'gold':gold, 'text':text.replace('  ', '\n')})

@app.route("/el")
def el():
    text = request.args.get("text", "")
    gold = json.loads(request.args.get("gold", "{}"))
    lang = request.args.get("lang", None)
    model = models[lang]

    return make_links(text, model.matcher, model.ned, gold=gold).replace("\n", "<br>")


