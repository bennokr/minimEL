# MinimEL: Minimalist Entity Linking

The `minimel` package provides a framework to create and evaluate small Entity Linking models.

> **Warning**
> This package is still under construction. A release is planned for the summer of 2023.

## App
To run the app, run `cd app` and then `flask run`.

## Evaluation Datasets

- [VoxEL: A Benchmark Dataset for Multilingual Entity Linking](https://figshare.com/articles/dataset/VoxEL/6539675)
- [Entity Linking in 100 Languages](https://github.com/google-research/google-research/tree/master/dense_representations_for_entity_retrieval/mel)
- [Tsai & Roth 2016](https://cogcomp.seas.upenn.edu/page/resource_view/102)

## IDEAS

- per surfaceform, ignore entities that are an instanceOf the top entity
- NER features
- global binary classifier with (ent feat, sent feat) tuples

## Tokenization, Stemming & Lemmatization
- Multi: https://github.com/mingruimingrui/ICU-tokenizer
- Multi: https://pypi.org/project/snowballstemmer/
- Japanese: https://github.com/SamuraiT/tinysegmenter
- Persian: https://github.com/htaghizadeh/PersianStemmer-Python
- Korean: https://pypi.org/project/soylemma/
