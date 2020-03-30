#!/usr/bin/env bash

## Generate bw_tables (best and worst ks for each classifier)
#MultinomNB Markov LinearSVC SK_Ovr_LR
#test_f1_weighted


### F1 score


#### CG
echo "CG"

./extract_best_worst_k.py\
    ../../data/outputs/HCV01_CG.json\
    ../../data/outputs/HCV02_CG.json\
    ../../data/outputs/bw_tables/HCV_CG_ALL_F1.tex\
    MultinomNB:Markov:SK_Ovr_LR:LinearSVC\
    test_f1_weighted\
    4:15

#### FT 100
echo "FT_100"

./extract_best_worst_k.py\
    ../../data/outputs/HCV01_FT_100_1000.json\
    ../../data/outputs/HCV02_FT_100_1000.json\
    ../../data/outputs/bw_tables/HCV_FT_100_ALL_F1.tex\
    MultinomNB:Markov:SK_Ovr_LR:LinearSVC\
    test_f1_weighted\
    4:15

#### FT_250
echo "FT_250"

./extract_best_worst_k.py\
    ../../data/outputs/HCV01_FT_250_1000.json\
    ../../data/outputs/HCV02_FT_250_1000.json\
    ../../data/outputs/bw_tables/HCV_FT_250_ALL_F1.tex\
    MultinomNB:Markov:SK_Ovr_LR:LinearSVC\
    test_f1_weighted\
    4:15

#### FT_500
echo "FT_500"

./extract_best_worst_k.py\
    ../../data/outputs/HCV01_FT_500_1000.json\
    ../../data/outputs/HCV02_FT_500_1000.json\
    ../../data/outputs/bw_tables/HCV_FT_500_ALL_F1.tex\
    MultinomNB:Markov:SK_Ovr_LR:LinearSVC\
    test_f1_weighted\
    4:15

#### FT_1000
echo "FT_1000"

./extract_best_worst_k.py\
    ../../data/outputs/HCV01_FT_1000_1000.json\
    ../../data/outputs/HCV02_FT_1000_1000.json\
    ../../data/outputs/bw_tables/HCV_FT_1000_ALL_F1.tex\
    MultinomNB:Markov:SK_Ovr_LR:LinearSVC\
    test_f1_weighted\
    4:15
