#!/usr/bin/env python

import sys
import json
import os.path
from collections import defaultdict
from pprint import pprint

import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Some dictionaries to fix text
# MultinomNB:Markov:LinearSVC:SK_Ovr_LR
algorithm = {"MultinomNB":"Multinomial Bayes", 
        "Markov":"Markov",
        "LinearSVC":"Linear SVM",
        "SK_Ovr_LR":"Logistic Regression" 
        }

model = { "MLE_MultinomNB":"MLE",
        "BAY_MultinomNB_Alpha_1e-100":"alpha=1e-100",
        "BAY_MultinomNB_Alpha_1e-10":"alpha=1e-10",
        "BAY_MultinomNB_Alpha_1e-5":"alpha=1e-5",
        "BAY_MultinomNB_Alpha_1e-2":"alpha=1e-2",
        "BAY_MultinomNB_Alpha_1":"alpha=1",

        "MLE_Markov":"MLE",
        "BAY_Markov_Alpha_1e-100":"alpha=1e-100",
        "BAY_Markov_Alpha_1e-10":"alpha=1e-10",
        "BAY_Markov_Alpha_1e-5":"alpha=1e-5",
        "BAY_Markov_Alpha_1e-2":"alpha=1e-2",
        "BAY_Markov_Alpha_1":"alpha=1",

        "SK_Ovr_LR_Liblinear_L1":"LR_L1",
        "SK_Ovr_LR_Liblinear_L2":"LR_L2",

        "SK_LinearSVC_SquaredHinge_L1_Primal":"LSVM_L1",
        "SK_LinearSVC_Hinge_L2_Dual":"hinge_L2_dual",
        "SK_LinearSVC_SquaredHinge_L2_Dual":"sqHinge_L2_dual",
        "SK_LinearSVC_SquaredHinge_L2_Primal":"LSVM_L2"
        }

def compile_data(json_sc, clf_kwds, kList, metric):
    scores_tmp = defaultdict(lambda : defaultdict(dict))
    scores = defaultdict(dict)

    for algo in json_sc:
        for kwd in clf_kwds: 
            if kwd in algo:
                if algo not in ("SK_LinearSVC_Hinge_L2_Dual", "SK_LinearSVC_SquaredHinge_L2_Dual"):
                    values =  [ np.array(json_sc[algo][str(k)][metric]) for k in kList ]

                    scores_tmp[kwd]["mean"][model[algo]] = np.array([ k.mean() for k in values ])
                    scores_tmp[kwd]["std"][model[algo]] = np.array([ k.std() for k in values ])

    #print(scores_tmp)
    for kwd in clf_kwds:
        scores[algorithm[kwd]]["mean"] = pd.DataFrame(scores_tmp[kwd]["mean"], columns=scores_tmp[kwd]["mean"].keys())
        scores[algorithm[kwd]]["mean"].index = kList
        scores[algorithm[kwd]]["std"] = pd.DataFrame(scores_tmp[kwd]["std"], columns=scores_tmp[kwd]["std"].keys())
        scores[algorithm[kwd]]["std"].index = kList

    #print(scores) 
    return scores


def compile_frgmt_data(score_dict, the_models):

    # "Multinomial Bayes": ["MLE", "alpha=1e-100"], 

    final_scores = defaultdict(lambda : defaultdict(dict))
    frgtm_columns = list(score_dict.keys())

    for frgmt_size in score_dict:
        # "Multinomial Bayes"
        for classifier in score_dict[frgmt_size]:
            if classifier in the_models:
                means = score_dict[frgmt_size][classifier]["mean"]
                stds = score_dict[frgmt_size][classifier]["std"]
                
                # "MLE"
                for model in the_models[classifier]:
                    if model not in final_scores[classifier]:
                        final_scores[classifier][model]["mean"] = pd.DataFrame(index=means.index, columns=frgtm_columns)
                        final_scores[classifier][model]["std"] = pd.DataFrame(index=stds.index, columns=frgtm_columns)

                    if model in means and model in stds:
                        final_scores[classifier][model]["mean"].loc[:, frgmt_size] = means[model]
                        final_scores[classifier][model]["std"][frgmt_size] = stds[model]

    #pprint(final_scores)
    return final_scores


def make_figure(scores, the_models, kList, out_file):
 
    #fig_format = "png"
    fig_format = "eps"
    fig_dpi = 150

    fig_file = out_file+"."+fig_format
    fig_title = os.path.splitext((os.path.basename(fig_file)))[0]
 
    cmap = cm.get_cmap('tab20')
    colors = [cmap(j/20) for j in range(0,20)] 
 
    styles = ["o-","^-","s-","x-","h-","d-","<-",">-","*-","p-"]
    sizefont = 12

    n_clfs = len(the_models)
    model_lists = the_models.values()
    n_models = len(list(the_models.values())[0])

    f, axs = plt.subplots(n_clfs, n_models, figsize=(15,12))
    axs = np.concatenate(axs)

    plt.rcParams.update({'font.size':sizefont})

    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    ind = 0
    for i, classifier in enumerate(the_models):

        for j, model in enumerate(scores[classifier]):
            p = scores[classifier][model]["mean"].plot(ax=axs[j+ind], style=styles, fontsize=sizefont)

            p.set_rasterization_zorder(0)

            for frgmt in scores[classifier][model]["mean"]:
                m = scores[classifier][model]["mean"][frgmt]
                s = scores[classifier][model]["std"][frgmt]

                p.fill_between(kList, m-s, m+s, alpha=0.1, zorder=-1)

            p.set_title(model)
            p.set_xticks([k for k in kList])
            p.set_xlim(min(kList)-0.5, max(kList)+0.5)
            p.set_ylim([0, 1.1])

            if i == n_clfs-1:
                p.set_xlabel('k-mers length', fontsize=sizefont+1)

            if j+ind != 1:
                p.get_legend().remove()

            else:
                p.legend(loc='upper left', fancybox=True, shadow=True, bbox_to_anchor=(1.01, 1.02))
            
            if j==0:
                p.set_ylabel(classifier, fontsize=sizefont+1)
            
            p.grid()
 
        ind +=2
    
    #plt.suptitle(fig_title)
    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)


if __name__ == "__main__":

    """
    ./make_typing_figure.py 
    ~/Projects/Thesis/Software/dna_bayes_clf/results/viruses/PR01/2019_06/
    HCV01
    test_plot
    test_f1_weighted 
    4:15
    """

    input_dir = sys.argv[1]
    virus = sys.argv[2]
    output_file = sys.argv[3]
    metric = sys.argv[4]
    str_k_list = sys.argv[5]

    models = {
            "Multinomial Bayes": ["MLE", "alpha=1e-100"],
            "Markov": ["MLE", "alpha=1e-100"], 
            "Logistic Regression": ["LR_L1", "LR_L2"],
            "Linear SVM": ["LSVM_L1", "LSVM_L2"]
            }

    clfs_list = list(algorithm.keys())
 
    suffix_files = ["_CG.json", "_FT_1000_1000.json", "_FT_500_1000.json", "_FT_250_1000.json", "_FT_100_1000.json"]
    fragments = ["CG", "1000 bp", "500 bp", "250 bp", "100 bp"]

    json_files = [os.path.join(input_dir, virus+sfx) for sfx in suffix_files]
    json_data = [json.load(open(jfile, "r")) for jfile in json_files  ]

    se_ks = str_k_list.split(":")
    
    if len(se_ks) != 2:
        print("K list argument should contain : to separate start and end")
        sys.exit()

    s_klen = int(se_ks[0])
    e_klen = int(se_ks[1])

    k_main_list = list(range(s_klen, e_klen+1))


    scores = {f:compile_data(j, clfs_list, k_main_list, metric) for f, j
            in zip(fragments, json_data)}
    
    frgmt_scores = compile_frgmt_data(scores, models)
 
    make_figure(frgmt_scores, models, k_main_list, output_file)

    with open(output_file+".txt", "w") as fh:
        pprint(frgmt_scores, fh)
