#!/usr/bin/env python

import sys
import json
import os.path
from collections import defaultdict
from pprint import pprint

import scipy.stats as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

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


def get_best_worst_ks(scores, clf_kwds, kList):
    
    results = defaultdict(lambda : defaultdict(dict))
    # mean and std dataframe have to have same index name array
    for algo_kwd in scores:
        mean_df = scores[algo_kwd]['mean']
        stds_df = scores[algo_kwd]['std']

        for model in mean_df:
            max_mean = mean_df[model].max()
            max_std = stds_df[model].loc[mean_df[model] == mean_df[model].max()].min()
            max_ks = mean_df[mean_df[model] == mean_df[model].max()].index.tolist()

            min_mean = mean_df[model].min()
            min_std = stds_df[model].loc[mean_df[model] == mean_df[model].min()].min()
            min_ks = mean_df[mean_df[model] == mean_df[model].min()].index.tolist()

            results[algo_kwd][model]['best_mean'] = max_mean
            results[algo_kwd][model]['best_std'] = max_std
            results[algo_kwd][model]['best_ks'] = max_ks

            results[algo_kwd][model]['worst_mean'] = min_mean
            results[algo_kwd][model]['worst_std'] = min_std
            results[algo_kwd][model]['worst_ks'] = min_ks

    #pprint(results)
    return results


def write_bw_scores(scores, out_file):
    header = r"""
    % \usepackage{booktabs}
    % \usepackage{multirow}
    \begin{table}[]
    \begin{tabular}{@{}llllll@{}}
    \toprule
    Classifier  & Model & \multicolumn{2}{l}{Best} & \multicolumn{2}{l}{Worst} \\ \midrule
                &       & Mean         & Ks        & Mean         & Ks         \\ \cmidrule(l){3-6}
    """ 
    
    footer = r"""

    \bottomrule
    \end{tabular}
    \end{table}
    """

    body = ""
    
    for i, algo_kwd in enumerate(scores):
        nb_models = len(scores[algo_kwd])
        multirow = "\\multirow{"+str(nb_models)+"}{*}{"+algo_kwd+"}"

        for j, model in enumerate(scores[algo_kwd]):

            max_mean = scores[algo_kwd][model]['best_mean']
            max_std = scores[algo_kwd][model]['best_std']
            max_ks = scores[algo_kwd][model]['best_ks']

            min_mean = scores[algo_kwd][model]['worst_mean']
            min_std = scores[algo_kwd][model]['worst_std']
            min_ks = scores[algo_kwd][model]['worst_ks']


            if j == 0: body += multirow

            body += "  & {} & {} \pm {} & {} & {} \pm {} & {} \\\\ ".\
                    format(model,
                            str(round(max_mean, 3)),
                            str(round(max_std, 3)), 
                            pprint_list(max_ks),
                            str(round(min_mean, 3)),
                            str(round(min_std, 3)),
                            pprint_list(min_ks))
            
            if i != len(scores)-1 and j == nb_models-1: body += " \\midrule"
            body += "\n"

    tex = header+body+footer
    with open(out_file, "w") as fout:
        fout.write(tex)


def write_couple_bw_scores(gen_scores, sub_scores, out_file):
    header = r"""
% \usepackage{booktabs}
% \usepackage{multirow}
\begin{table*}[]
\caption{}

\begin{tabularx}{\textwidth}{@{}llllllllll@{}}
\toprule
 &   & \multicolumn{4}{c}{Genotyping}  & \multicolumn{4}{c}{Subtyping}  \\ \cmidrule(l){3-10}
 &   & \multicolumn{2}{l}{Best} & \multicolumn{2}{l}{Worst} & \multicolumn{2}{l}{Best} & \multicolumn{2}{l}{Worst} \\ \cmidrule(l){3-10}
 Classifier & Model & F-measure & k lengths & F-measure  & k lengths  & F-measure & k lengths & F-measure & k lengths \\ \midrule
    """ 
    
    footer = r"""
\bottomrule

\end{tabularx}

\label{}
\end{table*}
    """

    body = ""
 
    for i, algo_kwd in enumerate(gen_scores):

        nb_models = len(gen_scores[algo_kwd])

        algo_tex = algo_kwd
        algo_split = algo_kwd.split(" ")
        if len(algo_split) > 1:
            algo_tex = "\\begin{tabular}[c]{@{}l@{}}"
                
            for z, chunk in enumerate(algo_split):
                algo_tex += chunk
                if z < len(algo_split)-1: algo_tex += " \\\\ "

            algo_tex +=" \\end{tabular} "

        multirow = " \\multirow{"+str(nb_models)+"}{*}{"+algo_tex+"}"

        for j, model in enumerate(gen_scores[algo_kwd]):

            max_mean_g = gen_scores[algo_kwd][model]['best_mean']
            max_std_g = gen_scores[algo_kwd][model]['best_std']
            max_ks_g = gen_scores[algo_kwd][model]['best_ks']

            min_mean_g = gen_scores[algo_kwd][model]['worst_mean']
            min_std_g = gen_scores[algo_kwd][model]['worst_std']
            min_ks_g = gen_scores[algo_kwd][model]['worst_ks']


            max_mean_s = sub_scores[algo_kwd][model]['best_mean']
            max_std_s = sub_scores[algo_kwd][model]['best_std']
            max_ks_s = sub_scores[algo_kwd][model]['best_ks']

            min_mean_s = sub_scores[algo_kwd][model]['worst_mean']
            min_std_s = sub_scores[algo_kwd][model]['worst_std']
            min_ks_s = sub_scores[algo_kwd][model]['worst_ks']

            if j == 0: body += multirow
            

            body += " & {} & {} $\pm$ {} & {} & {} $\pm$ {} & {} ".\
                    format(model.replace("_","\\_"),
                            str(round(max_mean_g, 3)),
                            str(round(max_std_g, 3)), 
                            pprint_list(max_ks_g),
                            str(round(min_mean_g, 3)),
                            str(round(min_std_g, 3)),
                            pprint_list(min_ks_g))

            body += " & {} $\pm$ {} & {} & {} $\pm$ {} & {} \\\\ ".\
                    format(str(round(max_mean_s, 3)),
                            str(round(max_std_s, 3)), 
                            pprint_list(max_ks_s),
                            str(round(min_mean_s, 3)),
                            str(round(min_std_s, 3)),
                            pprint_list(min_ks_s))
            
            if i != len(gen_scores)-1 and j == nb_models-1: body += " \\midrule"
            body += "\n"

    tex = header+body+footer
    with open(out_file, "w") as fout:
        fout.write(tex)


def pprint_list(liste):
    pile = []
    final = ""

    for elem in liste:
        if len(pile) > 0 and elem != pile[-1] +1:
            if len(pile) == 1:
                final += str(pile[0]) + ", "
            else:
                final += str(pile[0])+"-"+str(pile[-1])+", "
            pile = []

        pile.append(elem)
    
    if len(pile) == 1:
        final += str(pile[0])
    else:
        final += str(pile[0])+"-"+str(pile[-1])

    return final


if __name__ == "__main__":

    """
    ./extract_best_worst_k.py 
    ~/Projects/Thesis/dna_bayes_clf/results/viruses/PR01/2019_06/HCV01_CG.json
    test_table.tsv 
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

    geno_bw_scores = get_best_worst_ks(geno_scores, clfs_list, k_main_list)
    subt_bw_scores = get_best_worst_ks(subt_scores, clfs_list, k_main_list)

    #pprint(geno_bw_scores)
    #pprint(subt_bw_scores)
    #write_bw_scores(geno_bw_scores, output_file)
    write_couple_bw_scores(geno_bw_scores, subt_bw_scores, output_file) 

