#!/usr/bin/env python

from lm_genomvir.bio import seq_collections
from lm_genomvir.bio import kmer_collections as kmers
from lm_genomvir.utils import ndarrays_tolists

from lm_genomvir.bibm19 import clfs_to_evaluate

import warnings
#warnings.filterwarnings('ignore')

import sys
import json
import os.path
from pprint import pprint
from collections import defaultdict

import numpy as np

from joblib import Parallel, delayed

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


__author__ = "amine"


def clf_cross_val(classifier, X, X_conc, y, scoring, cv_iter,
        random_state, verbose):

    clf, use_X_back, clf_dscp = classifier
    final_X = X
    params = {}

    if use_X_back:
        final_X = X_conc
        params = {'v':X.shape[1]}

    skf = StratifiedKFold(n_splits=cv_iter, shuffle=True,
            random_state=random_state)


    scores_tmp = cross_validate(clf, final_X, y, scoring=scoring,
            cv=skf, fit_params=params, return_train_score=False)

    if verbose: print("Evaluated: {}\r".format(clf_dscp), flush=True)
    scores_tmp = ndarrays_tolists(scores_tmp)

    return clf_dscp, scores_tmp


def clfs_evaluation_mp(classifiers, X, X_b, y, cv_iter, scoring, n_proc=4,
        random_state=None, verbose=False):

    X_conc = np.concatenate((X, X_b), axis=1)
 
    if verbose: print("Cross-Validation step", flush=True)

    clf_cv_scores = Parallel(n_jobs=n_proc, prefer="processes")(
            delayed(clf_cross_val)(classifiers[clf_ind], X, X_conc, y,
                scoring, cv_iter, random_state, verbose)
            for clf_ind in classifiers)

    scores = {clf_dscp:cv_scores for clf_dscp, cv_scores in clf_cv_scores }

    return scores


def clfs_evaluation(classifiers, X, X_b, y, cv_iter, scoring,
        random_state=None, verbose=False):

    scores = dict()
    X_conc = np.concatenate((X, X_b), axis=1)
 
    if verbose: print("Cross-Validation step", flush=True)

    for clf_ind in classifiers:
        classifier, use_X_back, clf_dscp = classifiers[clf_ind]
        final_X = X
        params = {}

        if use_X_back:
            final_X = X_conc
            params = {'v':X.shape[1]}

        skf = StratifiedKFold(n_splits=cv_iter, shuffle=True,
                random_state=random_state)
        
        # cross_validate returns a dict of float arrays of shape (n_splits,)
        scores_tmp = cross_validate(classifier, final_X, y, scoring=scoring,
                cv=skf, n_jobs=-1, fit_params=params, return_train_score=False)

        if verbose: print("Evaluated: {}\r".format(clf_dscp), flush=True)

        scores[clf_dscp] = ndarrays_tolists(scores_tmp)

    return scores


def k_evaluation(seq_file, cls_file, classifiers, k_main_list, full_kmers,
        cv_iter, scoring, n_proc=4, random_state=None, verbose=True):

    k_scores = defaultdict(dict)
    
    if verbose: print("Dataset construction step", flush=True)

    seq_cv = seq_collections.SeqCollection((seq_file, cls_file))

    print("Counts of sequences")
    pprint(seq_cv.get_count_labels())

    for k_main in k_main_list:
    
        if verbose: print("\nProcessing k_main={}".format(k_main), flush=True)
 
        # # Data for cross validation
        seq_cv_kmers = kmers.build_kmers(seq_cv, k_main,
                full_kmers=full_kmers, sparse=None)
        seq_cv_X = seq_cv_kmers.data
        seq_cv_y = np.asarray(seq_cv.labels)

        seq_cv_back_kmers = kmers.build_kmers(seq_cv, k_main-1,
                full_kmers=full_kmers, sparse=None)
        seq_cv_X_back = seq_cv_back_kmers.data

        if n_proc == 0 :
            clf_scores = clfs_evaluation(classifiers, seq_cv_X,
                    seq_cv_X_back, seq_cv_y, cv_iter, scoring, 
                    random_state=random_state, verbose=verbose)
        else: 
            clf_scores = clfs_evaluation_mp(classifiers, seq_cv_X,
                    seq_cv_X_back, seq_cv_y, cv_iter, scoring, 
                    n_proc=n_proc, random_state=random_state, verbose=verbose)

        for clf_dscp in clf_scores:
            k_scores[clf_dscp][str(k_main)] = clf_scores[clf_dscp]

    return k_scores 


if __name__ == "__main__":
 
    """
    ./eval_complete_seqs.py\ 
    ../data/viruses/HIV01/data.fa\ 
    ../data/viruses/HIV01/class.csv\ 
    ../results/viruses/wcb_2019/tests/HIV01_mp.json\
    4 5 4 4
    """

    if len(sys.argv) != 9:
        print("8 arguments are needed!")
        sys.exit()

    print("RUN {}".format(sys.argv[0]), flush=True)

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]
    s_klen=int(sys.argv[4])
    e_klen=int(sys.argv[5])
    cv_iter = int(sys.argv[6])
    n_proc = int(sys.argv[7])
    clf_type = sys.argv[8]

    k_main_list = list(range(s_klen, e_klen+1))
    fullKmers = False
    eval_scores = ["recall_weighted", "precision_weighted", "f1_weighted"]
    rs = 0  # random_state
    verbose = True

    #if not os.path.isfile(scores_file):
    if True:
        the_scores = k_evaluation(seq_file, cls_file, clfs_to_evaluate[clf_type],
                k_main_list, fullKmers, cv_iter, eval_scores,
                n_proc=4, random_state=rs, verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    #pprint(the_scores)
