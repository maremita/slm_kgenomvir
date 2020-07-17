import slm_kgenomvir

import json
import platform

import numpy as np
import scipy
import sklearn
import Bio
import joblib
import matplotlib

__author__ = "amine"


def generate_all_words(alphabet, k):
    return ["".join(t) for t in product(alphabet, repeat=k)]


def ndarrays_tolists(obj):
    new_obj = dict()

    for key in obj:
        if isinstance(obj[key], np.ndarray):
            new_obj[key] = obj[key].tolist()

        else:
            new_obj[key] = obj[key]

    return new_obj


def rearrange_data_struct(data):
    new_data = defaultdict(dict)

    for k in data:
        for algo in data[k]:
            new_data[algo][k] = data[k][algo]

    return new_data


def load_Xy_data(xfile, yfile):

    X = scipy.sparse.load_npz(xfile)
    with open(yfile, 'r') as fh: y = np.array(json.load(fh))

    return X, y


def save_Xy_data(X, xfile, y, yfile):

    scipy.sparse.save_npz(xfile, X)
    with open(yfile, 'w') as fh: json.dump(y.tolist(), fh)


def get_ext_versions():
    versions = dict() 

    versions["python"] = platform.python_version()
    versions["slm_kgenomvir"] = slm_kgenomvir.__version__ 
    versions["numpy"] = np.__version__ 
    versions["scipy"] = scipy.__version__ 
    versions["sklearn"] = sklearn.__version__ 
    versions["Bio"] = Bio.__version__ 
    versions["joblib"] = joblib.__version__ 
    versions["matplotlib"] = matplotlib.__version__ 

    return versions
