import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from sklearn import datasets as skdatasets

# a list of the abbreviated name for all datasets
DS_NAMES = ["annthyroid", "cardio", "http", "mulcross", "musk",
                "pendigits", "satimage", "shuttle", "thyroid", "wbc"]

# the dimensions of each dataset
DS_DIMENSIONS = {
    'annthyroid': (7200, 6),
    'cardio': (1831, 21),
    'http': (567498, 3),
    'mulcross': (262144, 4),
    'musk': (3062, 166),
    'pendigits': (6870, 16),
    'satimage': (5803, 36),
    'shuttle': (49097, 9),
    'thyroid': (3772, 6),
    'wbc': (378, 30)
}


def load_data(dataset_name, normalize=True):
    '''
    Returns one of many possible datasets based on the given `dataset_name`

    Parameters
    ----------
    dataset_name : the abbreviated name of the dataset to load, see README for all datset options
    normalize    : if true, normalize the features of x to the range [0, 1]

    Returns
    -------
    x : the dataset samples with shape (nsamples, nfeatures)
    y : the dataset labels with shape (nsamples,) containing 0 to represent a normal label and 1 for an anomaly label
    '''

    if dataset_name == "annthyroid":
        return load_annthyroid(normalize)
    elif dataset_name == "cardio":
        return load_cardio(normalize)
    elif dataset_name == "http":
        return load_http(normalize)
    elif dataset_name == "mulcross":
        return load_mulcross(normalize)
    elif dataset_name == "musk":
        return load_musk(normalize)
    elif dataset_name == "pendigits":
        return load_pendigits(normalize)
    elif dataset_name == "satimage":
        return load_satimage(normalize)
    elif dataset_name == "shuttle":
        return load_shuttle(normalize)
    elif dataset_name == "thyroid":
        return load_thyroid(normalize)
    elif dataset_name == "wbc":
        return load_wbc(normalize)
    else:
        print("Unknown Dataset! Using thyroid")
        return load_thyroid(normalize)


def validate_labels(ds_names):
    '''
    Utility function for checking that all datset's labels are imported correctly. All datasetset should use 0 to represent normal, and 1 for anomalies and contain no other values in the returned y array

    Parameters
    ----------
    ds_names : a list of dataset names to evaluate e.g. ds_names = ["thyroid", "musk"] see README for abbreviated names

    Returns
    -------
    failed_ds : a list of dataset names which return labels other than 0 or 1
    '''
    failed_ds = []
    for ds in ds_names:
        x, y = load_data(ds)
        if not np.logical_or(y == 0, y == 1).all():
            failed_ds.append(ds)

    return failed_ds


def util_load_stonybrook(matpath, normalize=True):
    '''
    A utility method for loading any data matrix formatted as in the StonyBrook Outlier Detection Datasets (ODDS)
    '''
    data_dict = sio.loadmat(matpath)
    x = data_dict["X"]
    y = data_dict["y"].astype('int')

    # make y have dimension (nsamples,)
    y = np.squeeze(y)

    if normalize:
        # normalize x on [0, 1]
        max_value = np.max(x, axis=0)
        min_value = np.min(x, axis=0)
        x = (x - min_value) / (max_value - min_value)

    return x, y


def load_annthyroid(normalize=True):
    return util_load_stonybrook("./data/annthyroid/annthyroid.mat", normalize)


def load_cardio(normalize=True):
    return util_load_stonybrook("./data/cardio/cardio.mat", normalize)


def load_http(normalize=True):
    # Note the .mat file from StonyBrook ODDS is a version 7.3 mat file and not compataible with scipy, use HDF5 instead
    with h5py.File("./data/http/http.mat", "r") as f:
        x = f["X"][()].T
        y = f["y"][()].T.squeeze().astype('int')
    y = np.squeeze(y)

    if normalize:
        # normalize x on [0, 1]
        max_value = np.max(x, axis=0)
        min_value = np.min(x, axis=0)
        x = (x - min_value) / (max_value - min_value)

    return x, y


def load_mulcross(normalize=True):
    # load the dataset from a csv
    mulcross_df = pd.read_csv("./data/mulcross/mulcross.csv")

    # extract the feature columns to x
    x = mulcross_df[["V1", "V2", "V3", "V4"]].to_numpy()

    # convert text labels 'Normal' and 'Anomlay' to 0.0 and 1.0 respectively
    mulcross_df["y"] = 0
    idx_anomaly = mulcross_df["Target"] == "'Anomaly'"
    mulcross_df.loc[idx_anomaly, "y"] = 1
    y = mulcross_df["y"].to_numpy()

    return x, y


def load_musk(normalize=True):
    return util_load_stonybrook("./data/musk/musk.mat", normalize)


def load_pendigits(normalize=True):
    return util_load_stonybrook("./data/pendigits/pendigits.mat", normalize)


def load_satimage(normalize=True):
    return util_load_stonybrook("./data/satimage/satimage.mat", normalize)


def load_shuttle(normalize=True):
    return util_load_stonybrook("./data/shuttle/shuttle.mat", normalize)


def load_thyroid(normalize=True):
    return util_load_stonybrook("./data/thyroid/thyroid.mat", normalize)


def load_wbc(normalize=True):
    return util_load_stonybrook("./data/wbc/wbc.mat", normalize)


def load_iris(normalize=True):
    # import the iris dataset for testing
    iris = skdatasets.load_iris()
    x = iris.data
    y = iris.target

    # iris has three classes, lets use just two to simluate anomaly detection
    normal_class = 1
    anomal_class = 2

    idx_normal = (y == normal_class)
    idx_anomaly = (y == anomal_class)

    keep_idxs = np.append(idx_normal, idx_anomaly)

    x = x[keep_idxs]
    y = y[keep_idxs]

    if normalize:
        # normalize x on [0, 1]
        max_value = np.max(x, axis=0)
        min_value = np.min(x, axis=0)
        x = (x - min_value) / (max_value - min_value)

    return x, y
