import pandas
import numpy as np
import copy

import xgboost

import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import ShuffleSplit

import logging
import root_numpy as rnpy

import json

import resource

def log_memory():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
    logging.info("memory usage: {0:.2f} MB".format(mem))

ptbins = np.linspace(20, 620, 11)
ptbins2 = np.array([20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 200, 240, 280, 320, 360])

etabins = np.linspace(0, 2.5, 11)
etabins2 = np.linspace(0, 2.5, 11)

missing=-10

cols_replacevals = {
#    "Jet_CSV": {"replaceval_bad": -10, "replaceval_low": 0, "replaceval_high": 1},
#    "Jet_CSVIVF": {"replaceval_bad": -10, "replaceval_low": 0, "replaceval_high": 1},
#    "Jet_DeepCSVBDisc": {"replaceval_bad": -2, "replaceval_low": 0, "replaceval_high": 1},
    "Jet_cMVA": {"replaceval_bad": missing, "replaceval_low": -1, "replaceval_high": 1},
    "Jet_cMVAv2": {"replaceval_bad": missing, "replaceval_low": -1, "replaceval_high": 1},
    "Jet_JP": {"replaceval_bad": missing, "replaceval_low": 0, "replaceval_high": 10},
    "Jet_JBP": {"replaceval_bad": missing, "replaceval_low": 0, "replaceval_high": 30},
    "Jet_SoftMu": {"replaceval_bad": missing, "replaceval_low": 0, "replaceval_high": 1},
    "Jet_SoftEl": {"replaceval_bad": missing, "replaceval_low": 0, "replaceval_high": 1},
    "Jet_CSV": {"replaceval_bad": missing, "replaceval_low": 0, "replaceval_high": 1},
    "Jet_CSVIVF": {"replaceval_bad": missing, "replaceval_low": 0, "replaceval_high": 1},
    "Jet_DeepCSVBDisc": {"replaceval_bad": missing, "replaceval_low": 0, "replaceval_high": 1},
    
#    "Jet_cMVA": {"replaceval_bad": -1, "replaceval_low": -1, "replaceval_high": 1},
#    "Jet_cMVAv2": {"replaceval_bad": -1, "replaceval_low": -1, "replaceval_high": 1},
#    "Jet_JP": {"replaceval_bad": 0, "replaceval_low": 0, "replaceval_high": 10},
#    "Jet_JBP": {"replaceval_bad": 0, "replaceval_low": 0, "replaceval_high": 30},
#    "Jet_SoftMu": {"replaceval_bad": 0, "replaceval_low": 0, "replaceval_high": 1},
#    "Jet_SoftEl": {"replaceval_bad": 0, "replaceval_low": 0, "replaceval_high": 1},
#    "Jet_CSV": {"replaceval_bad": 0, "replaceval_low": 0, "replaceval_high": 1},
#    "Jet_CSVIVF": {"replaceval_bad": 0, "replaceval_low": 0, "replaceval_high": 1},
#    "Jet_DeepCSVBDisc": {"replaceval_bad": 0, "replaceval_low": 0, "replaceval_high": 1}
    
    
}

def normalize_col_inplace(data, col, showlog, replaceval_bad=0, replaceval_low=0, replaceval_high=1):
    """
   
    Args:
        data (TYPE): Description
        col (TYPE): Description
        replaceval_bad (int, optional): Description
        replaceval_low (int, optional): Description
        replaceval_high (int, optional): Description
    """
    if showlog==True:
        logging.debug("normalize_col_inplace {0}".format(col))
    vals = data[col]
    data.loc[np.isnan(vals), col] = replaceval_bad
    data.loc[np.isinf(vals), col] = replaceval_bad
    data.loc[vals<replaceval_low, col] = replaceval_bad  
    data.loc[vals>replaceval_high, col] = replaceval_high 
#    data.loc[vals==-2.0, col] = replaceval_bad 		#For DeepCSV
    return data

def normalize_all_cols(data, cols_replacevals,showlog):
    for col in data.columns:
        if cols_replacevals.has_key(col):
            normalize_col_inplace(data, col, showlog, **cols_replacevals[col])
    return data

def load_data(filename, treename, showlog=True, **kwargs):
    if showlog==True:
        logging.info("loading data from {0}:{1}".format(filename, treename))
	
    df = pandas.DataFrame(
        rnpy.root2array(
            filename,
            treename = treename,
            **kwargs
        )
    )
#    print df.head()
    if showlog==True:
        logging.info("loaded data with shape {0}".format(df.shape))
        log_memory()
    return df

def preprocess(data,showlog=True):#, train_frac=0.8):
    if showlog==True:
        logging.info("preprocessing data")

    #Here we categorize the data according to the jet pdgId into 4 main groups
    data["flavour_category"] = 0
    data.loc[data["Jet_flavour"].abs() <= 3, "flavour_category"] = 0
    data.loc[data["Jet_flavour"].abs() == 4, "flavour_category"] = 1
    data.loc[data["Jet_flavour"].abs() == 5, "flavour_category"] = 2
    data.loc[data["Jet_flavour"].abs() == 21, "flavour_category"] = 0

    data["is_training"] = data["index"] <= 4
    data["weight"] = 1.0

    data["ptbin"] = map(lambda x: ptbins.searchsorted(x), data["Jet_pt"])
    data["ptbin2"] = map(lambda x: ptbins2.searchsorted(x), data["Jet_pt"])
    data["etabin"] = map(lambda x: etabins.searchsorted(x), data["Jet_eta"].abs())
    data["etabin2"] = map(lambda x: etabins2.searchsorted(x), data["Jet_eta"].abs())

    normalize_all_cols(data, cols_replacevals,showlog)

    return data

def filter_data(data, expr):
    return data[data.eval(expr)]

def calc_weight_binary(data, target):
    sig = data[target] == 1
    bkg = data[target] == 0
    data.loc[sig, "weight"] = 1.0 / np.sum(sig)
    data.loc[bkg, "weight"] = 1.0 / np.sum(bkg)
    return data

def classifier(cls, kwargs):
    return cls(**kwargs)

def fit_classifier(cls, data, variables, target):
    cls.fit(
        data[variables].as_matrix(),
        data[target].as_matrix(),
        sample_weight=data["weight"].ravel()
    )
    return cls

def make_tuple(*args):
    return tuple(args)

def get_weights(df, ptbins, etabins):
    """Given a data frame, calculate the pt-eta weights
    to make the kinematic distributions flat.
    
    Args:
        df (DataFrame): jets
        ptbins (array): Bins in pt
        etabins (array): Bins in eta
    
    Returns:
        2d array: weight distribution
    """
    vs, bx, by = np.histogram2d(
        df["Jet_pt"],
        df["Jet_eta"].abs(),
        bins=[ptbins, etabins]
    )
    #normalize
    vs = vs / float(np.sum(vs))
    #invert
    ws = 1.0/vs
    #get rid of bad values
    ws[np.isnan(ws)]=0.0
    ws[np.isinf(ws)]=1
    return ws

def train_crossvalidation(cls, data_tuple):
    rets = []
    models = []
    kf = ShuffleSplit(n_splits = 5)
    X = data_tuple[0].as_matrix()
    y = data_tuple[1].as_matrix()
    for train, test in kf.split(X):
        logging.info("Training crossvalidation {0}:{1}".format(len(train), len(test)))
        fitted = cls.fit(
            X[train],
            y[train],
        )
        models += [fitted]
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            y[test],
            fitted.predict_proba(X[test])[:, 1]
        )
        rets += [(tpr, fpr)]
    return rets

def train_variable_remove(cls, data_tuple_train, data_tuple_test):
    cols = data_tuple_train[0].columns
    rets = []
    varsets = []
    for c in cols:
        new_cols = copy.deepcopy(list(cols))
        new_cols.pop(new_cols.index(c))
        logging.info("Training variable remove with -{0}".format(c))
        cls.fit(
            data_tuple_train[0][new_cols],
            data_tuple_train[1],
            verbose = False,
        )
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            data_tuple_test[1],
            cls.predict_proba(data_tuple_test[0][new_cols])[:, 1]
        )
        rets += [(tpr, fpr)]
        varsets += [c]
    return rets, varsets

def save_set(eval_set, fn):
    logging.info("save_set to {0}".format(fn))
    i = 0
    data = {}
    for a, b in eval_set:
        logging.info("features {0}, target {1}".format(a.shape, b.shape))
        fna = "{0}_{1}.svmbin".format(fn, i)
        fi = open(fna, "wb")
        sklearn.datasets.dump_svmlight_file(a, b, fi)
        fi.close()
        data["eval_set_{0}".format(i)] = fna
        i += 1
    fi = open(fn + ".json", "w")
    json.dump(data, fi)
    fi.close()

def load_set(fn):
    print "load_set"
    fi = open(fn + ".json")
    data = json.load(fi)
    fi.close()
    eval_set = []
    for k in sorted(data.keys()):
        eval_set += [
            xgboost.DMatrix(
                data[k]
                #+"#cache{0}".format(k)
            )
        ]
    return eval_set
