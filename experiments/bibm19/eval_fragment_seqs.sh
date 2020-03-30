#!/bin/bash

# author: amine
# An example bash script to run eval_fragment_seqs.py


CLF=lr
VIRUS=HCV02
CGFRGT=FT
FRGTSIZE=100
FRGTNB=1000

NBITER=4
NBPROC=0

VSET=${VIRUS}_${CGFRGT}_${FRGTSIZE}_${FRGTNB}_${CLF}

SK=8
EK=8


OUTDIR=../../data/outputs/$VSET

FASTA=../../data/viruses/$VIRUS/data.fa
CLASS=../../data/viruses/$VIRUS/class.csv

echo "Classification with Fragments ${VSET}"

if [ ! -d $OUTDIR ]; then
    mkdir -p $OUTDIR
fi

JSONF=$OUTDIR/${VSET}_${SK}_${EK}.json
./eval_fragment_seqs.py $FASTA $CLASS $JSONF $SK $EK $FRGTSIZE $FRGTNB $NBITER $NBPROC $CLF
