from flask import Flask, request
import dawg

lang_trie = {
    'nl': '../data/nlwiki-20211120.count.min2.salient.completiondawg',
    'simple': '../data/simplewiki-20211120.count.min2.salient.completiondawg',
}

app = Flask(__name__)

@app.route('/')
def hello_world():
    return """
<!DOCTYPE html>
<html>
<head><title>Minimal EL demo</title></head>
<style>
body { font-family: sans-serif; }
#main {margin: 0 auto; max-width: 800px;}
textarea { width: 100%; height: 8em }
</style>
<script type="text/javascript">
function link() {
    text = document.getElementById('text').value;
    lang = document.getElementById('lang').value;
    document.getElementById('spinner').style.visibility = 'visible';

    fetch('el?' + new URLSearchParams({text: text, lang:lang}))
    .then(response => response.text()).then(data => {
        document.getElementById('result').innerHTML = data;
        document.getElementById('spinner').style.visibility = 'hidden';
    })
}
</script>
<body>
<div id="main">
    <h1>Minimal EL hello world</h1>
    <p>
        <select id="lang">
            <option value=nl>Nederlands</option>
            <option value=simple>Simple English</option>
        </select>
    </p>
    <p><textarea id="text" placeholder="Your text here"></textarea></p>
    <p><button onclick="link()">Link</button><img src="static/load.gif" id="spinner" style="height:1em; visibility:hidden; "/></p>
    <p id="result"></p>
</div>
</body>
</html>
    """

def get_matches(surface_trie, text):
    normtoks = text.lower().split()
    for i,tok in enumerate(normtoks):
        for comp in surface_trie.keys(tok):
            comp_toks = comp.lower().split()
            if normtoks[i:i+len(comp_toks)] == comp_toks:
                yield i, comp

def make_links(surface_trie, text):
    normtoks = text.split()
    matches = {}
    for i,m in get_matches(surface_trie, text):
        matches.setdefault(i, set()).add(m)

    offset, out = 0, ""
    for i,m in sorted(matches.items()):
        if i >= offset:
            comp = max(m, key=len).split()
            out += ' '.join(normtoks[offset:i])
            w = ' '.join(normtoks[i:i+len(comp)])
            out += f' <a href="#{i}">{w}</a> '
            offset = i + len(comp)
    out += ' '.join(normtoks[offset:])
    return out


@app.route('/el')
def el():
    text = request.args.get('text', '')
    lang = request.args.get('lang', None)

    ftrie = lang_trie[lang]

    surface_trie = dawg.CompletionDAWG()
    surface_trie.load(ftrie)

    return make_links(surface_trie, text).replace('\n', '<br>')
