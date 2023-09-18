import os, logging

from flask import Flask, request, render_template

import minimel
from minimel.mentions import get_matches, setup_matcher
import app_deploy

app = Flask(minimel.__name__, root_path='../app/')

app.add_url_rule('/update_server', methods=['POST'], view_func=app_deploy.update)


models = {
    # "nl": "Nederlands",
    "simple": "Simple English",
}
app.logger.info(f'Loading matchers...')
lang_matcher = {
    # "nl": setup_matcher("../wiki/nlwiki-20220301/count.min2.json"),
    "simple": setup_matcher("../wiki/simplewiki-20211120/count.min2.json"),
}
basedir = os.path.dirname(os.path.realpath(__file__))


@app.route("/")
def index():
    return render_template('index.html', 
        models=models,
    )


def make_links(matcher, text):
    matches = {}
    for start, m in get_matches(matcher, text.lower()):
        matches.setdefault(start, set()).add(m)

    offset, out = 0, ""
    for start, pos_matches in sorted(matches.items()):
        if start >= offset:
            comp = max(pos_matches, key=len)
            out += text[offset:start]
            w = text[start : start + len(comp)]
            out += f'<a href="#{start}">{w}</a>'
            offset = start + len(comp)
    out += text[offset:]
    return out


@app.route("/el")
def el():
    text = request.args.get("text", "")
    lang = request.args.get("lang", None)

    matcher = lang_matcher[lang]

    return make_links(matcher, text).replace("\n", "<br>")


