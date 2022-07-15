import subprocess
import sys
import os
import time
import requests
import atexit
import tqdm
import dawg
import json
import re

JARFILE = 'dbpedia-spotlight-model/rest/target/rest-1.1-jar-with-dependencies.jar'
PORT = '2222'

def get_annotations(url, t):
    r = requests.get(url,
                     params={'text':t}, 
                     headers={'Accept':'application/json'})
    if r.ok:
        return r.json()

if __name__ == '__main__':
    lang = sys.argv[1]
    dawgfile = sys.argv[2]
    fname = sys.argv[3]
    
    index = dawg.IntDAWG()
    index.load(str(dawgfile))
    inlines = open(fname).read().splitlines()        
    
    url = f'http://0.0.0.0:{PORT}/rest'
    p = subprocess.Popen([
            'java','-Dfile.encoding=UTF-8','-Xmx10G',
            '-jar', JARFILE,
            lang,
            url
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    atexit.register(lambda: p.kill())
    
    while True:
        time.sleep(1)
        try:
            if get_annotations( url+ '/annotate', inlines[0].split('\t')[-1] ):
                break
        except requests.exceptions.ConnectionError:
            pass

    for line in tqdm.tqdm(inlines):
        i, ents, text = line.split('\t')
        text = text.replace('"','\\"')
        xml = f'<annotation text="{text}">\n'
        for e in json.loads(ents):
            try:
                m = re.search(e, text)
                if m:
                    e = e.replace('"','\\"')
                    xml += f'<surfaceForm name="{e}" offset="{m.start()}" />\n'
            except re.error: continue
        xml += '</annotation>'
        ents = {}
        annot = get_annotations( url+ '/disambiguate', xml ) or {}
        for rec in annot.get('Resources', []):
            uri = rec.get('@URI', '')
            sf = rec.get('@surfaceForm', '')
            e = index.get(re.sub('http://.*dbpedia.org/resource/','', uri))
            if e:
                ents[sf] = e
        print(i, json.dumps(ents), sep='\t')