#!/usr/bin/env bash

## Generate figures
#MultinomNB Markov LinearSVC SK_Ovr_LR
#test_f1_weighted


### F1 score


#### CG
./make_typing_figure.py\
    ../../data/outputs/HCV01_CG.json\
    ../../data/outputs/HCV02_CG.json\
    ../../data/outputs/figures/HCV_CG_BAY_MKV_F1\
    MultinomNB:Markov\
    test_f1_weighted\
    4:15

./make_typing_figure.py\
    ../../data/outputs/HCV01_CG.json\
    ../../data/outputs/HCV02_CG.json\
    ../../data/outputs/figures/HCV_CG_LR_SVM_F1\
    SK_Ovr_LR:LinearSVC\
    test_f1_weighted\
    4:15

#### FT 100
#./make_typing_figure.py ../../data/outputs/HCV01_FT_100_1000.json ../../data/outputs/HCV02_FT_100_1000.json ../../data/outputs/figures/HCV_FT_100_BAY_MKV_F1 MultinomNB:Markov test_f1_weighted 4:15


#./make_typing_figure.py ../../data/outputs/HCV01_FT_100_1000.json ../../data/outputs/HCV02_FT_100_1000.json ../../data/outputs/figures/HCV_FT_100_LR_SVM_F1 SK_Ovr_LR:LinearSVC test_f1_weighted 4:15

#./make_typing_figure.py ../../data/outputs/HCV01_FT_100_1000.json ../../data/outputs/HCV02_FT_100_1000.json ../../data/outputs/figures/HCV_FT_100_ALL_F1 MultinomNB:Markov:SK_Ovr_LR:LinearSVC test_f1_weighted 4:15

#### FT_250
#./make_typing_figure.py ../../data/outputs/HCV01_FT_250_1000.json ../../data/outputs/HCV02_FT_250_1000.json ../../data/outputs/figures/HCV_FT_250_BAY_MKV_F1 MultinomNB:Markov test_f1_weighted 4:15


#./make_typing_figure.py ../../data/outputs/HCV01_FT_250_1000.json ../../data/outputs/HCV02_FT_250_1000.json ../../data/outputs/figures/HCV_FT_250_LR_SVM_F1 SK_Ovr_LR:LinearSVC test_f1_weighted 4:15

#./make_typing_figure.py ../../data/outputs/HCV01_FT_250_1000.json ../../data/outputs/HCV02_FT_250_1000.json ../../data/outputs/figures/HCV_FT_250_ALL_F1 MultinomNB:Markov:SK_Ovr_LR:LinearSVC test_f1_weighted 4:15

#### FT_500
#./make_typing_figure.py ../../data/outputs/HCV01_FT_500_1000.json ../../data/outputs/HCV02_FT_500_1000.json ../../data/outputs/figures/HCV_FT_500_BAY_MKV_F1 MultinomNB:Markov test_f1_weighted 4:15


#./make_typing_figure.py ../../data/outputs/HCV01_FT_500_1000.json ../../data/outputs/HCV02_FT_500_1000.json ../../data/outputs/figures/HCV_FT_500_LR_SVM_F1 SK_Ovr_LR:LinearSVC test_f1_weighted 4:15

#./make_typing_figure.py ../../data/outputs/HCV01_FT_500_1000.json ../../data/outputs/HCV02_FT_500_1000.json ../../data/outputs/figures/HCV_FT_500_ALL_F1 MultinomNB:Markov:SK_Ovr_LR:LinearSVC test_f1_weighted 4:15

#### FT_1000
#./make_typing_figure.py ../../data/outputs/HCV01_FT_1000_1000.json ../../data/outputs/HCV02_FT_1000_1000.json ../../data/outputs/figures/HCV_FT_1000_BAY_MKV_F1 MultinomNB:Markov test_f1_weighted 4:15


#./make_typing_figure.py ../../data/outputs/HCV01_FT_1000_1000.json ../../data/outputs/HCV02_FT_1000_1000.json ../../data/outputs/figures/HCV_FT_1000_LR_SVM_F1 SK_Ovr_LR:LinearSVC test_f1_weighted 4:15

#./make_typing_figure.py ../../data/outputs/HCV01_FT_1000_1000.json ../../data/outputs/HCV02_FT_1000_1000.json ../../data/outputs/figures/HCV_FT_1000_ALL_F1 MultinomNB:Markov:SK_Ovr_LR:LinearSVC test_f1_weighted 4:15
