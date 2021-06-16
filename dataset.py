import numpy as np
import scipy.io as sio
import os
from sklearn import datasets as skdatasets


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
    y : the dataset labels with shape (nsamples,)
    '''
    if dataset_name == "thyroid":
        return load_thyroid(normalize)
    elif dataset_name == "wbc":
        return load_wbc(normalize)
    elif dataset_name == "cardio":
        return load_cardio(normalize)
    elif dataset_name == "musk":
        return load_musk(normalize)
    else:
        print("Unknown Dataset! Using thyroid")


def util_load_stonybrook(matpath, normalize=True):
    '''
    A utility method for loading any data matrix formatted as in the StonyBrook Outlier Detection Datasets (ODDS)
    '''
    data_dict = sio.loadmat(matpath)
    x = data_dict["X"]
    y = data_dict["y"]

    normal_class = 0
    anomaly_class = 1

    # make anomaly label -1, normal label 1
    idx_normal = (y == normal_class)
    idx_anomaly = (y == anomaly_class)

    y[idx_normal] = 1
    y[idx_anomaly] = -1

    # make y have dimension (nsamples,)
    y = np.squeeze(y)

    if normalize:
        # normalize x on [0, 1]
        max_value = np.max(x, axis=0)
        min_value = np.min(x, axis=0)
        x = (x - min_value) / (max_value - min_value)

    return x, y


def load_thyroid(normalize=True):
    return util_load_stonybrook("./data/thyroid/thyroid.mat", normalize)


def load_wbc(normalize=True):
    return util_load_stonybrook("./data/wbc/wbc.mat", normalize)


def load_cardio(normalize=True):
    return util_load_stonybrook("./data/cardio/cardio.mat", normalize)


def load_musk(normalize=True):
    return util_load_stonybrook("./data/musk/musk.mat", normalize)


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

    y[idx_normal] = 1
    y[idx_anomaly] = -1

    keep_idxs = np.append(idx_normal, idx_anomaly)

    x = x[keep_idxs]
    y = y[keep_idxs]

    if normalize:
        # normalize x on [0, 1]
        max_value = np.max(x, axis=0)
        min_value = np.min(x, axis=0)
        x = (x - min_value) / (max_value - min_value)

    return x, y
