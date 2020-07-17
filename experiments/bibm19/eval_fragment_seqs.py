#!/usr/bin/env python

from slm_kgenomvir.bio import seq_collections
from slm_kgenomvir.bio import kmer_collections as kmers
from slm_kgenomvir.bibm19 import clfs_to_evaluate

import sys
import json
import os.path
from collections import defaultdict
import time
from pprint import pprint

import numpy as np
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import recall_score, precision_score, f1_score


__author__ = "amine"


def clf_cross_val(classifier, X_train, X_conc_train, y_train,
        X_test, X_conc_test, y_test, scoring, verbose=True):

    clf, use_X_back, clf_dscp = classifier

    new_clf = clone(clf)

    final_X_train = X_train 
    final_X_test = X_test
    params = dict()
    scores_tmp = dict()
    average="weighted" 

    if use_X_back:
        final_X_train = X_conc_train
        final_X_test = X_conc_test
        params = {'v':X_train.shape[1]}

    # train step
    start = time.time()
    new_clf.fit(final_X_train, y_train, **params)
    end = time.time()
    scores_tmp["fit_time"] = end - start

    # test step
    start = time.time()
    y_pred = new_clf.predict(final_X_test)
    end = time.time() 
    scores_tmp["score_time"] = end - start

    if verbose: print("Evaluated: {}\r".format(clf_dscp), flush=True)

    for scorer_name in scoring:
        scorer = scoring[scorer_name]
        scores_tmp["test_{}_{}".format(scorer_name, average)] = scorer(
                    y_test, y_pred, average=average)

    return clf_dscp, scores_tmp


def clfs_evaluation_mp(classifiers, X_train, X_conc_train, y_train,
        X_test, X_conc_test, y_test, scoring, n_proc, verbose=True):
  
    clf_cv_scores = Parallel(n_jobs=n_proc, prefer="processes")(
            delayed(clf_cross_val)(classifiers[clf_ind], X_train, 
                X_conc_train, y_train, X_test, X_conc_test, y_test,
                scoring, verbose) for clf_ind in classifiers)

    scores = {clf_dscp:scores for clf_dscp, scores in clf_cv_scores}

    return scores


def clfs_evaluation(classifiers, X_train, X_conc_train, y_train,
        X_test, X_conc_test, y_test, scoring, verbose=True):

    scores = dict()

    for clf_ind in classifiers:
        classifier, use_X_back, clf_dscp = classifiers[clf_ind]

        new_clf = clone(classifier)

        final_X_train = X_train 
        final_X_test = X_test
        params = dict()
        scores_tmp = dict()
        average="weighted"

        if use_X_back:
            final_X_train = X_conc_train
            final_X_test = X_conc_test
            params = {'v':X_train.shape[1]}

        # train step
        start = time.time()
        new_clf.fit(final_X_train, y_train, **params)
        end = time.time()
        scores_tmp["fit_time"] = end - start

        # test step
        start = time.time()
        y_pred = new_clf.predict(final_X_test)
        end = time.time() 
        scores_tmp["score_time"] = end - start

        if verbose: print("Evaluated: {}\r".format(clf_dscp), flush=True)

        for scorer_name in scoring:
            scorer = scoring[scorer_name]
            scores_tmp["test_{}_{}".format(scorer_name, average)] = scorer(
                    y_test, y_pred, average=average)

        scores[clf_dscp] = scores_tmp

    return scores


def clf_evaluation_with_fragments(classifiers, data_seqs, fragments,
        parents, k, full_kmers, nb_iter, scoring, n_proc,
        random_state=None, verbose=False):

    #scores_iter = defaultdict(lambda: [0]*nb_iter)
    scores_iter = defaultdict(dict)
    #final_scores = dict()
    final_scores = defaultdict(lambda: defaultdict(list))

    if verbose: print("Validation step", flush=True)
 
    ## construct X and X_back dataset
    X_kmer = kmers.build_kmers(data_seqs, k, full_kmers=full_kmers,
            sparse=None)
    X = X_kmer.data
    #print("X shape {}".format(X.shape))
    X_kmers_list = X_kmer.kmers_list
    y = np.asarray(data_seqs.labels)

    # X_back
    X_kmer_back = kmers.build_kmers(data_seqs, k-1,
            full_kmers=full_kmers, sparse=None)
    X_back = X_kmer_back.data
    #print("X_back shape {}".format(X_back.shape))
    X_back_list = X_kmer_back.kmers_list

    ## construct fragments dataset
    X_frgmts = kmers.GivenKmersCollection(fragments, X_kmers_list,
            sparse=None).data
    #print("X_frgmts shape {}".format(X_frgmts.shape))
    X_frgmts_back = kmers.GivenKmersCollection(fragments, X_back_list,
            sparse=None).data    
    y_frgmts = np.asarray(fragments.labels)
    #print("X_frgmts_back shape {}".format(X_frgmts_back.shape))
    #print("y_frgmts shape {}".format(y_frgmts.shape))

    seq_ind = list(i for i in range(0,len(data_seqs)))
    sss = StratifiedShuffleSplit(n_splits=nb_iter, test_size=0.2,
            random_state=random_state)

    for ind_iter, (train_ind, test_ind) in enumerate(sss.split(seq_ind, y)):
        if verbose: print("\nIteration {}\n".format(ind_iter), flush=True)

        X_train = X[train_ind]
        X_back_train = X_back[train_ind]
        X_conc_train = np.concatenate((X_train, X_back_train), axis=1)
        y_train = y[train_ind] 

        # Get fragments test indices
        ind_frgmts = np.array([i for p in test_ind for i in parents[p]])
 
        X_test = X_frgmts[ind_frgmts]
        X_back_test = X_frgmts_back[ind_frgmts]
        X_conc_test = np.concatenate((X_test, X_back_test), axis=1)
        y_test = y_frgmts[ind_frgmts]

        #print("X_test shape {}".format(X_test.shape))
        #print("y_test shape {}".format(y_test.shape))
       
        if n_proc == 0 :
            clf_scores = clfs_evaluation(classifiers, X_train, X_conc_train, y_train,
                X_test, X_conc_test, y_test, scoring, verbose=verbose)

        else: 
            clf_scores = clfs_evaluation_mp(classifiers, X_train, X_conc_train, y_train,
                X_test, X_conc_test, y_test, scoring, n_proc, verbose=verbose)

        for clf_dscp in clf_scores:
            scores_iter[clf_dscp][ind_iter] = clf_scores[clf_dscp]

    for clf_dscp in scores_iter:
        for n_iter in scores_iter[clf_dscp]:
            for score_ in scores_iter[clf_dscp][n_iter]:
                final_scores[clf_dscp][score_].append(scores_iter[clf_dscp][n_iter][score_])

    return final_scores


def k_evaluation_with_fragments(seq_file, cls_file, classifiers,
        k_main_list, full_kmers, frgmt_size, frgmt_count, nb_iter,
        scoring, n_proc, random_state=None, verbose=True):

    k_scores = defaultdict(dict)

    if verbose: print("Dataset construction step", flush=True)

    seq_cv = seq_collections.SeqCollection((seq_file, cls_file))

    print("Counts of complete genomes")
    pprint(seq_cv.get_count_labels())

    ## construct fragments dataset
    frgmts_cv = seq_cv.extract_fragments(frgmt_size, stride=int(frgmt_size/2))

    if frgmt_count > 1:
        # TODO:  with value of 1, I got an error in line 175
        frgmts_cv = frgmts_cv.stratified_sample(frgmt_count)

    print("Counts of fragments")
    pprint(frgmts_cv.get_count_labels())

    frgmts_parents = frgmts_cv.get_parents_rank_list()

    for k_main in k_main_list:
        if verbose: print("\nProcessing k_main={}".format(k_main), flush=True)

        clf_scores = clf_evaluation_with_fragments(classifiers,
                seq_cv, frgmts_cv, frgmts_parents, k_main, full_kmers, 
                nb_iter, scoring, n_proc, random_state=random_state, verbose=verbose)

        for clf_dscp in clf_scores:
            k_scores[clf_dscp][str(k_main)] = clf_scores[clf_dscp]

    return k_scores


if __name__ == "__main__":
 
    """
    ./eval_fragment_seqs.py\ 
    ../data/viruses/HIV01/data.fa\ 
    ../data/viruses/HIV01/class.csv\ 
    ../results/viruses/wcb_2019/tests/HIV01_frg_mp.json\
    4 5 1000 5 4
    """
    
    if len(sys.argv) != 11:
        print("10 arguments are needed!")
        sys.exit()

    print("RUN {}".format(sys.argv[0]), flush=True)

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]
    s_klen=int(sys.argv[4])
    e_klen=int(sys.argv[5])
    fragment_size = int(sys.argv[6])
    fragment_count = int(sys.argv[7])
    nb_iter = int(sys.argv[8])
    n_proc = int(sys.argv[9])
    clf_type = sys.argv[10]

    k_main_list = list(range(s_klen, e_klen+1))
    full_kmers = False
    eval_scores = {"recall":recall_score, 
            "precision":precision_score, 
            "f1":f1_score}

    rs = 0  # random_state
    verbose = True

    #if not os.path.isfile(scores_file):
    if True:
        the_scores = k_evaluation_with_fragments(seq_file, cls_file,
                clfs_to_evaluate[clf_type], k_main_list, full_kmers,
                fragment_size, fragment_count, nb_iter, eval_scores,
                n_proc, random_state=rs, verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    #pprint(the_scores)
