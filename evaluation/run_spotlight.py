import subprocess
import sys
import time
import requests
import atexit
import tqdm
import dawg
import json
import re
import html

JARFILE = "dbpedia-spotlight-model/rest/target/rest-1.1-jar-with-dependencies.jar"
PORT = "2223"


def get_annotations(url, t, timeout=None):
    r = requests.post(
        url, data={"text": t}, headers={"Accept": "application/json"}, timeout=timeout
    )
    if r.ok:
        return r.json()


if __name__ == "__main__":
    fname = sys.argv[1]
    inlines = open(fname).read().splitlines()
    lang = sys.argv[2]

    dawgfile = sys.argv[3]

    index = dawg.IntDAWG()
    index.load(str(dawgfile))

    url = f"http://0.0.0.0:{PORT}/rest"
    p = subprocess.Popen(
        ["java", "-Dfile.encoding=UTF-8", "-Xmx10G", "-jar", JARFILE, lang, url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    atexit.register(lambda: p.kill())

    while True:
        time.sleep(1)
        try:
            test = html.escape(inlines[0].split("\t")[-1])
            if get_annotations(url + "/annotate", test):
                break
        except requests.exceptions.ConnectionError:
            pass

    for line in tqdm.tqdm(inlines):
        i, ents, text = line.split("\t")
        xml = f'<annotation text="{html.escape(text)}">\n'
        for e in json.loads(ents):
            e = e.replace("!", "")
            try:
                m = re.search(e, text)
                if m:
                    e = html.escape(e)
                    xml += f'<surfaceForm name="{e}"    offset="{m.start()}" />\n'
            except re.error:
                continue
        xml += "</annotation>"
        ents = {}
        try:
            annot = get_annotations(url + "/disambiguate", xml, timeout=5) or {}
            for rec in annot.get("Resources", []):
                uri = rec.get("@URI", "")
                sf = rec.get("@surfaceForm", "")
                e = index.get(re.sub("http://.*dbpedia.org/resource/", "", uri))
                if e:
                    ents[sf] = e
            print(i, json.dumps(ents), sep="\t")
        except requests.exceptions.ReadTimeout:
            print(i, "{}", sep="\t")
        sys.stdout.flush()
