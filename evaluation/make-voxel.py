# https://figshare.com/articles/dataset/VoxEL/6539675
import os
import subprocess
import glob
import rdflib
import dawg
import tqdm
import json
import logging

def load_annotations(fname, wm):
    g = rdflib.Graph().load(fname, format='turtle')
    nif = rdflib.Namespace('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#')
    itsrdf = rdflib.Namespace('http://www.w3.org/2005/11/its/rdf#')

    paragraph_annotations = {}
    for annot, context in g[:nif.Context]:
        paragraph = str(g.value(context, nif.isString))
        source = str(g.value(context, nif.sourceUrl))
        anchor = str(g.value(annot, nif.anchorOf))
        ref = str(g.value(annot, itsrdf.taIdentRef))
        ref = wm.get( ref.split('wikipedia.org/wiki/')[-1] )
        if ref:
            para = paragraph_annotations.setdefault((source, paragraph), set())
            para.add( (anchor, ref ) )
    
    return paragraph_annotations

def run(*cmd, cwd=None):
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, cwd=cwd
    ) as p:
        for line in p.stdout:
            print(line, end='')
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, p.args))

if __name__ == '__main__':
    
    datasets = ['en','de','fr','it','es']

    if not os.path.exists('VoxEL-ttl'):
        os.makedirs('VoxEL-ttl')
        run('wget','https://figshare.com/ndownloader/files/12012098','-O','VoxEL-ttl/VoxEL.zip')
        run('unzip','VoxEL-ttl/VoxEL.zip','-d','VoxEL-ttl')

    if not os.path.exists('VoxEL'):
        os.makedirs('VoxEL')
        for lang in datasets:
            data_file = f'VoxEL-ttl/sVoxEL-{lang}.ttl'
            for index in glob.glob(f'../wiki/{lang}wiki-*/index_{lang}wiki-*.dawg'):
                logging.info(f'Using {index}')
                wm = dawg.IntDAWG().load(index)

                with open(f'VoxEL/{lang}.tsv', 'w') as fw:
                    annotations = load_annotations(data_file, wm)
                    for (code, text), ents in annotations.items():
                        ents = json.dumps(dict(ents))
                        print(str(code), ents, text, sep='\t', file=fw)
                break