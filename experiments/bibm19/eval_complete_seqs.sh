#!/bin/bash

# author: amine
# An example bash script to run eval_complete_seqs.py

CLF=markov
VIRUS=HCV02
CGFRGT=CC
NBITER=4
NBPROC=0

VSET=${VIRUS}_${CGFRGT}_${CLF}

SK=4
EK=4

OUTDIR=../../data/outputs/$VSET

FASTA=../../data/viruses/$VIRUS/data.fa
CLASS=../../data/viruses/$VIRUS/class.csv

echo "Classification with Complete genomes ${VIRUS}"

if [ ! -d $OUTDIR ]; then
    mkdir -p $OUTDIR
fi

JSONF=$OUTDIR/${VSET}_${SK}_${EK}.json
./eval_complete_seqs.py $FASTA $CLASS $JSONF $SK $EK $NBITER $NBPROC $CLF
