import pathlib, logging

from flask import Flask, request, render_template

import minimel
from minimel.mentions import get_matches, setup_matcher

import app_deploy

app = Flask(__name__)
app.add_url_rule('/update_server', methods=['POST'], view_func=app_deploy.update)


models = {
    # "nl": "Nederlands",
    "simple": "Simple English",
}
app.logger.info(f'Loading matchers...')
basedir = pathlib.Path(__file__).parent.parent
lang_matcher = {
    # "nl": setup_matcher("../wiki/nlwiki-20220301/count.min2.json"),
    "simple": setup_matcher(basedir / "wiki/simplewiki-20211120/count.min2.json"),
}

app.logger.info(f'Loading NED models...')
lang_ned = {
    "simple": minimel.MiniNED(
        basedir / "wiki/simplewiki-20211120/index_simplewiki-20211120.dawg"
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
            out += f'<a href="https://www.wikidata.org/wiki/Q{link}">{w}</a>'
            offset = start + len(comp)
    out += text[offset:]
    return out


@app.route("/el")
def el():
    text = request.args.get("text", "")
    lang = request.args.get("lang", None)

    matcher = lang_matcher[lang]
    ned = lang_ned[lang]

    return make_links(text, matcher, ned).replace("\n", "<br>")


