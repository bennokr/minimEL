# MinimEL: Minimalist Entity Linking

The `minimel` package provides a framework to create and evaluate small Entity Linking models.

## Evaluation Datasets

- [VoxEL: A Benchmark Dataset for Multilingual Entity Linking](https://figshare.com/articles/dataset/VoxEL/6539675)
- [Entity Linking in 100 Languages](https://github.com/google-research/google-research/tree/master/dense_representations_for_entity_retrieval/mel)


## TODO

- spotlight baseline
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