# Minimalist Entity Linking



## Evaluation Datasets

- [VoxEL: A Benchmark Dataset for Multilingual Entity Linking](https://figshare.com/articles/dataset/VoxEL/6539675)
- [Entity Linking in 100 Languages](https://github.com/google-research/google-research/tree/master/dense_representations_for_entity_retrieval/mel)

## Usage

1. Make Wikipedia article title -> Wikidata ID mapping
```
wikimapper download $WIKI-$VERSION
wikimapper create $WIKI-$VERSION
minimel index index_$WIKI-$VERSION.db
```
2. Get list-anchors from disambiguation pages
```
wget https://dumps.wikimedia.org/$WIKI/$VERSION/$WIKI-$VERSION-pages-articles.xml.bz2
bunzip2 $VERSION-pages-articles.xml.bz2
minimel get-disambig $WIKI-$VERSION-pages-articles.xml index_$WIKI-$VERSION.dawg disambig_page_ids.txt
```
3. Create wikipedia paragraph links dataset
```
minimel -sv get-paragraphs $WIKI-$VERSION-pages-articles.xml index_$WIKI-$VERSION.dawg
```


4. Count surface-link occurrences
```
minimel -sv count $WIKI-$VERSION-paragraph-links/
minimel -sv count $WIKI-$VERSION-paragraph-links/ -l $LANGCODE
```


5. Filter out bad surface-link candidates
```
for q in "0.25" "0.5" "1"; do 
    for s in "" "-stem"; do 
        prun -np 1 minimel clean index_$WIKI-$VERSION.db disambig.json count.min2$s.json \
            -b ../minimal-EL/data/wikidata-20211122-disambig.txt \
            -s $q ${s:+-l $LANGCODE} \
            </dev/null > clean$s-q$q.json & 
    done; 
done
```
(move & rename these files)

make entity features

6. Create training dataset 
```
for q in "0.25" "0.5" "1"; do 
    for s in "" "-stem"; do 
        for f in "" "feat-clean-q1.p5"; do 
            minimel -sv vectorize ../$WIKI-$VERSION-paragraph-links/ clean$s-q$q.json \
                ${f:+--ent-feats-csv $f.csv} ${s:+-l $LANGCODE}; 
            cat clean$s-q$q${f:+.$f}.parts/* > clean$s-q$q${f:+.$f}.dat && 
            rm -r clean$s-q$q${f:+.$f}.parts; 
        done; 
    done; 
done
```

7. Train models
```
for b in "16" "20" "24"; do 
    for q in "0.25" "0.5" "1"; do 
        for s in "" "-stem"; do 
            for f in "" "feat-clean-q1.p5"; do 
                fname=train$s-q$q${f:+.$f}-b$b; 
                echo $fname; 
                prun -v -np 1 -t 24:00:00 time minimel -v train clean$s-q$q${f:+.$f}.dat \
                    -b $b </dev/null &> $fname.log & 
            done; 
        done; 
    done; 
done
```

8. Run models
```
for b in "16" "20" "24"; do 
    for q in "0.25" "0.5" "1"; do 
        for s in "" "-stem"; do 
            for f in "" "feat-clean-q1.p5"; do 
                minimel -v run -p  ../index_$WIKI-$VERSION.dawg clean-q$q.json \
                    clean-q$q${f:+.$f}."$b"b.vw \
                    ${f:+--ent-feats-csv $f.csv} ${s:+-l $LANGCODE} \
                    <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv \
                    > pred-mewsli${s:-'-clean'}-q$q${f:+.$f}."$b"b.tsv; 
                minimel -v run -p  ../index_$WIKI-$VERSION.dawg clean-q$q.json \
                    clean-q$q${f:+.$f}."$b"b.vw -c clean$s-q1.json \
                    ${f:+--ent-feats-csv $f.csv} ${s:+-l $LANGCODE} \
                    <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv \
                    > pred-mewsli${s:-'-clean'}-q$q${f:+.$f}."$b"b.clean$s-q1.tsv; 
            done; 
        done; 
    done; 
done

minimel -v run -p ../index_$WIKI-$VERSION.dawg <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv > pred-mewsli-base.tsv; minimel -v run -p ../index_$WIKI-$VERSION.dawg -c clean-q1.json <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv > pred-mewsli-base.clean-q1.tsv; 
minimel -v run -p ../index_$WIKI-$VERSION.dawg -c clean-stem-q1.json -l $LANGCODE  <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv > pred-mewsli-base.clean-stem-q1.tsv

minimel -v run -p ../index_$WIKI-$VERSION.dawg -c clean-q1.json -u <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv > pred-mewsli-upper.clean-q1.tsv; 
minimel -v run -p ../index_$WIKI-$VERSION.dawg -c clean-stem-q1.json -l $LANGCODE -u <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv > pred-mewsli-upper.clean-stem-q1.tsv
```


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