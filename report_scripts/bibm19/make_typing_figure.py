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


def make_figure(scores1, scores2, clf_kwds, kList, metric, out_file):
 
    #fig_format = "png"
    fig_format = "eps"
    fig_dpi = 150

    fig_file = out_file+"."+fig_format
    fig_title = os.path.splitext((os.path.basename(fig_file)))[0]
 
    cmap = cm.get_cmap('tab20')
    colors = [cmap(j/20) for j in range(0,20)] 
 
    styles = ["o-","^-","s-","x-","h-","d-","<-",">-","*-","p-"]
    sizefont = 12

    f, axs = plt.subplots(len(clf_kwds), 2, figsize=(15,8))
    axs = np.concatenate(axs)

    plt.rcParams.update({'font.size':sizefont})

    plt.subplots_adjust(wspace=0.1)

    ind = 0
    for algo_kwd in scores1:

        h1 = scores1[algo_kwd]["mean"].plot(ax=axs[ind], style=styles, fontsize=sizefont)
        h2 = scores2[algo_kwd]["mean"].plot(ax=axs[ind+1], style=styles, fontsize=sizefont)

        # For ESP transparent rendering
        h1.set_rasterization_zorder(0)
        h2.set_rasterization_zorder(0)

        if ind > 1:
            h1.set_xlabel('k-mers length', fontsize=sizefont+1)
            h2.set_xlabel('k-mers length', fontsize=sizefont+1)

        if ind < 2:
            h1.set_title('Genotyping (HCVGENCG)')
            h2.set_title('Subtyping (HCVSUBCG)')

        h1.set_ylabel(algo_kwd, fontsize=sizefont+1)

        h1.set_xticks([k for k in kList])
        h2.set_xticks([k for k in kList])

        h1.set_xlim(min(kList)-0.5, max(kList)+0.5)
        h2.set_xlim(min(kList)-0.5, max(kList)+0.5)
        
        h1.set_ylim([0, 1.1])
        h2.set_ylim([0, 1.1])
        
        h1.get_legend().remove()
 
        if ind < 2 or len(scores1) > 2:
            h2.legend(loc='upper left', fancybox=True, shadow=True, 
                    bbox_to_anchor=(1.01, 1.02))
        else:
            h2.get_legend().remove()

        for algo in scores1[algo_kwd]["mean"]:
            m = scores1[algo_kwd]["mean"][algo]
            s = scores1[algo_kwd]["std"][algo]

            h1.fill_between(kList, m-s, m+s, alpha=0.1, zorder=-1)

            m = scores2[algo_kwd]["mean"][algo]
            s = scores2[algo_kwd]["std"][algo]

            h2.fill_between(kList, m-s, m+s, alpha=0.1, zorder=-1)

        h1.grid()
        h2.grid()

        ind +=2
    
    #plt.suptitle(fig_title)
    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)


if __name__ == "__main__":

    """
    ./make_typing_figure.py 
    ~/Projects/Thesis/Software/dna_bayes_clf/results/viruses/PR01/2019_06/HCV01_CG.json
    ~/Projects/Thesis/Software/dna_bayes_clf/results/viruses/PR01/2019_06/HCV02_CG.json 
    test_plot 
    MultinomNB:Markov 
    test_f1_weighted 
    4:15
    """

    geno_file = sys.argv[1]
    subt_file = sys.argv[2]
    output_file = sys.argv[3]
    clf_keyword = sys.argv[4]
    metric = sys.argv[5]
    str_k_list = sys.argv[6]

    clfs_list = clf_keyword.split(":")

    for clf in clfs_list:
        if clf not in algorithm:
            print("Classifier keywords should be one of these :")
            print(algorithm.keys())
            sys.exit()

    se_ks = str_k_list.split(":")
    if len(se_ks) != 2:
        print("K list argument should contain : to separate start and end")
        sys.exit()

    s_klen = int(se_ks[0])
    e_klen = int(se_ks[1])

    k_main_list = list(range(s_klen, e_klen+1))
 
    geno_json = json.load(open(geno_file, "r"))
    subt_json = json.load(open(subt_file, "r"))

    geno_scores = compile_data(geno_json, clfs_list, k_main_list, metric)
    subt_scores = compile_data(subt_json, clfs_list, k_main_list, metric)

    make_figure(geno_scores, subt_scores, clfs_list, k_main_list, metric, output_file)

    with open(output_file+".txt", "w") as fh:
        fh.write("Genotype results: \n\n")
        pprint(geno_scores, fh)
        fh.write("\nSubtype results: \n\n")
        pprint(subt_scores, fh)
