# https://github.com/google-research/google-research/tree/master/dense_representations_for_entity_retrieval/mel
import os
import sys
import subprocess
import glob
import tqdm
import json
import logging
import pandas as pd

def load_annotations(lang):
    mentions = pd.read_csv(root+'/'+lang+'/mentions.tsv', sep='\t')
    if 'qid' in mentions:
        mentions['qid'] = mentions['qid'].str[1:].astype('int')
    else:
        import dawg
        
        for index in glob.glob(f'../wiki/{lang}wiki-*/index_{lang}wiki-*.dawg'):
            logging.info(f'Using {index}')
            wm = dawg.IntDAWG().load(index)
        mentions['qid'] = mentions['url'].apply(lambda x: wm.get(x[29:], -1))
        mentions = mentions[mentions['qid'] > 0]
                
    docs = {
        docid: open(root+'/'+lang+'/text/'+docid).read() 
        for docid in tqdm.tqdm(set(mentions['docid']), 'Loading documents', ncols=80)
    }
    links = mentions.set_index('docid')[['mention','qid']].sort_index()
    paragraph_annotations = {
        (docid, doc.replace('\n', ' ')): [tuple(a) for a in links.loc[[docid]].values]
        for docid, doc in tqdm.tqdm(docs.items(), desc='Collecting annotations', ncols=80)
    }
    return paragraph_annotations

def run(*cmd, cwd=None):
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, cwd=cwd
    ) as p:
        for line in p.stdout:
            print(line, end='')
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, p.args)

if __name__ == '__main__':
    
    datasets = ['en','ar','de','es','fa','ja','sr','ta','tr', 'nl']
    root = 'dense_representations_for_entity_retrieval/mel/mewsli-9/output/dataset/'
    url = 'https://github.com/google-research/google-research/trunk/dense_representations_for_entity_retrieval'

    if not os.path.exists(root):
        os.makedirs(root)
        run('svn','export','url')
        run('bash','get-mewsli-9.sh', 
            cwd='dense_representations_for_entity_retrieval/mel')
    
    
    bad_ents = set()
    for fname in sys.argv[1:]:
        bad_ents |= set(int(l) for l in open(fname))
    print(f'Filtering out {len(bad_ents)} bad entities')
    
    if not os.path.exists('Mewsli-9'):
        os.makedirs('Mewsli-9')
    for lang in datasets:
        print(f'Making {lang}')
        annotations = load_annotations(lang)

        with open(f'Mewsli-9/{lang}.tsv', 'w') as fw:
            for (code, text), ents in annotations.items():
                ents = json.dumps({s:e for s,e in ents if e not in bad_ents})
                print(str(code), ents, text, sep='\t', file=fw)