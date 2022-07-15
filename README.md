# Minimal Entity Linking Baseline



## Evaluation Datasets

- [VoxEL: A Benchmark Dataset for Multilingual Entity Linking](https://figshare.com/articles/dataset/VoxEL/6539675)
- [Entity Linking in 100 Languages](https://github.com/google-research/google-research/tree/master/dense_representations_for_entity_retrieval/mel)

## Usage

1. Make Wikipedia article title -> Wikidata ID mapping
```
wikimapper download $WIKIDUMP-$VERSION
wikimapper create $WIKIDUMP-$VERSION
minimel index $WIKIDUMP-$VERSION.db
```
2. Get list-anchors from disambiguation pages
```
wget https://dumps.wikimedia.org/$WIKIDUMP/$VERSION/$WIKIDUMP-$VERSION-pages-articles.xml.bz2
bunzip2 $VERSION-pages-articles.xml.bz2
minimel get-disambig $WIKIDUMP-$VERSION-pages-articles.xml $WIKIDUMP-$VERSION.dawg disambig_page_ids.txt
```
3. Create wikipedia paragraph dataset

4. Count surface-link occurrences
5. Filter out bad surface-link candidates
6. Create training dataset 
7. Train models

## TODO

- spotlight baseline
- performance per candidate size
- stopword removal


## IDEAS

- per surfaceform, ignore entities that are an instanceOf the top entity
- NER features
- global binary classifier with (ent feat, sent feat) tuples

## Tokenization, Stemming & Lemmatization
- Multi: https://github.com/mingruimingrui/ICU-tokenizer
- Multi: https://pypi.org/project/snowballstemmer/
- Japanese: https://github.com/SamuraiT/tinysegmenter
- Korean: https://pypi.org/project/soylemma/
# minimEL
