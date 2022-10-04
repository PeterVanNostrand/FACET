import numpy as np
import pandas as pd
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
