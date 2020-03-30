#!/usr/bin/env python

from os.path import isfile
import sys
import json
from collections import defaultdict
from glob import glob


if __name__ == "__main__":
    """
    This script merges the json files contained in one or several folders 
    according to the models and the ks.
    This is useful when distributing the evaluation scripts by models and ks

    Command line:

    merge_models_ks_jsons.py <input directory> [output file] 

    The path of input files needs to be absolute

    If the path does not end by a "/", the script add a */* to get all
    the folders that match this pattern, otherwise (ends with "/"), 
    it lists only the files in this file

    """

    k_scores = defaultdict(dict)
    final_scores = defaultdict(dict)
    
    if len(sys.argv) < 2:
        print("The script needs at least one argument, the input directory")
        sys.exit()

    my_path = sys.argv[1]
    output_file = None

    # output file
    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    
    if output_file is None:
        if my_path.endswith("/"):
            output_file = my_path.rstrip("/")+".json"
        else:
            output_file = my_path+".json"

    # Get files
    suffix = "*/*"
    if my_path.endswith("/"):
        suffix = "/*"

    files = [f for f in  glob(my_path+suffix) if isfile(f) and
            f.endswith(".json") ]

    for one_file in files:
        scores = json.load(open(one_file, "r"))

        for clf_dscp in scores:
            for str_k in scores[clf_dscp]:
                k_scores[clf_dscp][str_k] = scores[clf_dscp][str_k]
    
    # Sort Ks
    for clf_dscp in k_scores:
        for str_k in sorted(k_scores[clf_dscp], key=lambda v: int(v)):
            final_scores[clf_dscp][str_k] = k_scores[clf_dscp][str_k]

    with open(output_file ,"w") as fh_out: 
        json.dump(final_scores, fh_out, indent=2)
