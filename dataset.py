import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from sklearn import datasets as skdatasets
from sklearn.preprocessing import StandardScaler

# a list of the abbreviated name for all classification datasets
DS_NAMES = [
    "cancer",
    "glass",
    "magic",
    "spambase",
    "vertebral"
]

DS_PATHS = {
    "cancer": "data/cancer/wdbc.data",
    "glass": "data/glass/glass.data",
    "magic": "data/magic/magic04.data",
    "spambase": "data/spambase/spambase.data",
    "vertebral": "data/vertebral/column_2C.dat"
}

DS_DIMENSIONS = {
    "cancer": (568, 30),
    "glass": (162, 9),
    "magic": (19019, 10),
    "spambase": (4600, 57),
    "vertebral": (309, 6)
}

# a list of the abbreviated name for all anomaly detection datasets
ANOM_DS_NAMES = ["annthyroid", "cardio", "http", "mulcross", "musk",
                 "pendigits", "satimage", "shuttle", "thyroid", "wbc"]

# the dimensions of each anomly detection dataset
ANOM_DS_DIMENSIONS = {
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

ANOM_DS_PATHS = {
    "annthyroid": "data/anomaly-detection/annthyroid/annthyroid.mat",
    "cardio": "data/anomaly-detection/cardio/cardio.mat",
    "http": "data/anomaly-detection/http/http.mat",
    "mulcross": "data/anomaly-detection/mulcross/mulcross.csv",
    "musk": "data/anomaly-detection/musk/musk.mat",
    "pendigits": "data/anomaly-detection/pendigits/pendigits.mat",
    "satimage": "data/anomaly-detection/satimage/satimage.mat",
    "shuttle": "data/anomaly-detection/shuttle/shuttle.mat",
    "thyroid": "data/anomaly-detection/thyroid/thyroid.mat",
    "wbc": "data/anomaly-detection/wbc/wbc.mat"
}


def load_data(dataset_name, preprocessing: str = "Normalize"):
    '''
    Returns one of many possible anomaly detetion datasets based on the given `dataset_name`

    Parameters
    ----------
    dataset_name : the abbreviated name of the dataset to load, see README for all datset options
    preprocessing: if None do nothing, if Normalize normalize all features [0,1], if Scale Standardize features by removing the mean and scaling to unit variance

    Returns
    -------
    x : the dataset samples with shape (nsamples, nfeatures)
    y : the dataset labels with shape (nsamples,) containing 0 to represent a normal label and 1 for an anomaly label
    '''
    if dataset_name == "cancer":
        x, y = util_load_cancer()
    elif dataset_name == "glass":
        x, y = util_load_glass()
    elif dataset_name == "magic":
        x, y = util_load_magic()
    elif dataset_name == "spambase":
        x, y = util_load_spambase()
    elif dataset_name == "vertebral":
        x, y = util_load_vertebral()
    else:
        print("ERROR NO SUCH DATASET")
        exit(0)

    if preprocessing == "Normalize":
        # normalize x to [0, 1]
        max_value = np.max(x, axis=0)
        min_value = np.min(x, axis=0)
        x = (x - min_value) / (max_value - min_value)
    elif preprocessing == "Scale":
        float_transformer = StandardScaler()
        x = float_transformer.fit_transform(x)
    return x, y


def util_load_cancer():
    data = pd.read_csv(DS_PATHS["cancer"]).to_numpy()
    ids = data[:, 0:1]
    labels = data[:, 1:2].squeeze()  # lables of M=Malignant, B=Benign

    # label malignant as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "M"] = 1
    x = data[:, 2:].astype(float)
    return x, y


def util_load_glass():
    data = pd.read_csv(DS_PATHS["glass"]).to_numpy()
    ids = data[:, 0:1]
    labels = data[:, -1:].squeeze()

    # glass has 6 classes which correspond to six types of glass
    # 1 building_windows_float_processed
    # 2 building_windows_non_float_processed
    # 3 vehicle_windows_float_processed
    # 4 vehicle_windows_non_float_processed (none in this database)
    # 5 containers
    # 6 tableware
    # 7 headlamps
    # We aim to train a binary classifier to differentiate between float and nonfloat glass
    float_classes = [1, 3]
    nonfloat_classes = [2, 4]

    is_float = np.isin(labels, float_classes)
    is_nonfloat = np.isin(labels, nonfloat_classes)

    # label float as 1, nonfloat as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[is_float] = 1

    x = data[:, 1:-1].astype(float)

    # drop all samples that are not float or nonfloat (keeps ~75% of dataset)
    y = y[is_float | is_nonfloat]
    x = x[is_float | is_nonfloat]
    return x, y


def util_load_magic():
    data = pd.read_csv(DS_PATHS["magic"]).to_numpy()
    labels = data[:, -1:].squeeze()  # lables of g=gamma (signal), h=hadron (background)

    # label gamma (signal) as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "g"] = 1
    x = data[:, :-1].astype(float)
    return x, y


def util_load_spambase():
    path = DS_PATHS["spambase"]
    data = pd.read_csv(path).to_numpy()
    ncols = data.shape[1]
    x = data[:, :-1]
    y = data[:, -1:].squeeze().astype('int')
    return x, y


def util_load_vertebral():
    data = pd.read_csv(DS_PATHS["vertebral"], delim_whitespace=True).to_numpy()
    labels = data[:, -1:].squeeze()  # lables of g=gamma (signal), h=hadron (background)

    # label gamma (signal) as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "AB"] = 1
    x = data[:, :-1].astype(float)
    return x, y


def load_anom_data(dataset_name, preprocessing="Scale"):
    '''
    Returns one of many possible anomaly detetion datasets based on the given `dataset_name`

    Parameters
    ----------
    dataset_name : the abbreviated name of the dataset to load, see README for all datset options
    preprocessing: if None do nothing, if Normalize normalize all features [0,1], if Scale Standardize features by removing the mean and scaling to unit variance

    Returns
    -------
    x : the dataset samples with shape (nsamples, nfeatures)
    y : the dataset labels with shape (nsamples,) containing 0 to represent a normal label and 1 for an anomaly label
    '''

    ds_odds_format = ["annthyroid", "cardio", "musk", "pendigits", "satimage", "shuttle", "thyroid", "wbc"]
    if dataset_name in ds_odds_format:
        x, y = util_load_stonybrook(dataset_name)
    elif dataset_name == "http":
        x, y = load_http()
    elif dataset_name == "mulcross":
        x, y = load_mulcross()
    else:
        print("Unknown Dataset! Using thyroid")
        x, y = util_load_stonybrook("thyroid", )

    if preprocessing == "Normalize":
        # normalize x to [0, 1]
        # x = (x - min_value) / (max_value - min_value)
        x = normalize(x)
    elif preprocessing == "Scale":
        float_transformer = StandardScaler()
        x = float_transformer.fit_transform(x)

    return x, y


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


def util_load_stonybrook(dataset_name, ):
    '''
    A utility method for loading any data matrix formatted as in the StonyBrook Outlier Detection Datasets (ODDS)
    '''
    matpath = ANOM_DS_PATHS[dataset_name]
    data_dict = sio.loadmat(matpath)
    x = data_dict["X"]
    y = data_dict["y"].astype('int')
    # make y have dimension (nsamples,)
    y = np.squeeze(y)
    return x, y


def load_http():
    # Note the .mat file from StonyBrook ODDS is a version 7.3 mat file and not compataible with scipy, use HDF5 instead
    with h5py.File("./data/http/http.mat", "r") as f:
        x = f["X"][()].T
        y = f["y"][()].T.squeeze().astype('int')
    y = np.squeeze(y)
    return x, y


def load_mulcross():
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


def load_iris():
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
    return x, y
