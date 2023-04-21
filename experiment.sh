#!/usr/bin/env bash
set -Eeuo pipefail

dumps="eswiki 20220301 es"

# dumps="simplewiki 20211120 en
# tawiki 20220301 ta
# trwiki 20220301 tr
# arwiki 20220301 ar
# srwiki 20220301 sr
# jawiki 20220301 ja
# dewiki 20220301 de"

# echo "$dumps" | while IFS= read -r dump; do
#     read WIKI VERSION LANGCODE <<< $dump
#     cd wiki/$WIKI-$VERSION/experiments/
#     for b in "28"; do 
#         for q in "0.25" "0.5" "1"; do 
#             for s in "" "-stem"; do 
#                 # "feat-clean-q1.p5"
#                 for f in ""; do
#                     fname=train$s-q$q${f:+.$f}-b$b; 
#                     echo $(pwd)"/"$fname; 
#                     prun -v -np 1 -t 24:00:00 time minimel -v train clean$s-q$q${f:+.$f}.dat -b $b </dev/null &> $fname.log &
#                 done; 
#             done;
#         done;
#     done;
#     cd -;
# done;

echo "$dumps" | while IFS= read -r dump; do
    read WIKI VERSION LANGCODE <<< $dump
    cd wiki/$WIKI-$VERSION/experiments/
    for b in "16" "20" "24"; do 
        for q in "1"; do # "0.25" "0.5" 
            for s in ""; do # "-stem"
                for f in ""; do 
                    prun -v -np 1 -t 24:00:00 minimel -v run -p  ../index_$WIKI-$VERSION.dawg clean-q$q.json clean-q$q${f:+.$f}."$b"b.vw ${f:+--ent-feats-csv $f.csv} ${s:+-l $LANGCODE} <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv  > pred-mewsli${s:-'-clean'}-q$q${f:+.$f}."$b"b.tsv & 
                    prun -v -np 1 -t 24:00:00 minimel -v run -p  ../index_$WIKI-$VERSION.dawg clean-q$q.json clean-q$q${f:+.$f}."$b"b.vw -c clean$s-q1.json ${f:+--ent-feats-csv $f.csv} ${s:+-l $LANGCODE} <  ../../minimal-EL/evaluation/Mewsli-9/$LANGCODE.tsv  > pred-mewsli${s:-'-clean'}-q$q${f:+.$f}."$b"b.clean$s-q1.tsv & 
                done;
            done; 
        done; 
    done;
    cd -;
done;
