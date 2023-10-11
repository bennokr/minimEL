import pathlib
from dataclasses import dataclass

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


def make_links(text, matcher, ned):
    matches = {}
    for start, m in get_matches(matcher, text.lower()):
        matches.setdefault(start, set()).add(m)

    offset, out = 0, ""
    for start, pos_matches in sorted(matches.items()):
        if start >= offset:
            comp = max(pos_matches, key=len)
            out += text[offset:start]
            w = text[start : start + len(comp)]
            link = ned.predict(text, w)
            if link:
                out += f'<a href="https://www.wikidata.org/wiki/Q{link}">{w}</a>'
            else:
                out += w
            offset = start + len(comp)
    out += text[offset:]
    return out

@app.route("/random")
def random():
    from random import choice
    lang = request.args.get("lang", None)
    model = models[lang]
    _, _, texts = zip(*(l.split('\t') for l in open(model.eval)))
    return choice(texts).replace('  ', '\n')

@app.route("/el")
def el():
    text = request.args.get("text", "")
    lang = request.args.get("lang", None)
    model = models[lang]

    return make_links(text, model.matcher, model.ned).replace("\n", "<br>")


