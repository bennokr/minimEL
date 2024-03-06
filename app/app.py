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
    matches = {}
    for start, m in get_matches(matcher, text.lower()):
        matches.setdefault(start, set()).add(m)

    if gold:
        gold_matcher = setup_matcher(None, names=list(gold))
        gold_matches = {}
        for start, m in get_matches(gold_matcher, text):
            gold_matches.setdefault(start, {})[m] = gold[m]

    offset, out = 0, ""
    positions = sorted(set(matches) | set(gold_matches))
    for start in positions:
        if start >= offset and start in matches:
            matched = max(matches[start], key=len)
            out += text[offset:start]
            w = text[start : start + len(matched)]
            link = ned.predict(text, w)
            if link:
                out += f'<a href="https://www.wikidata.org/wiki/Q{link}">{w}</a>'
            else:
                out += w
            offset = start + len(matched)
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


