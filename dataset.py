import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from config import DO_VIZUALIZATION, VIZ_DATA_PATH

if DO_VIZUALIZATION:
    from visualization.viz_tools import save_data_dict

script_directory = os.path.dirname(os.path.abspath(__file__))

# a list of the abbreviated name for all classification datasets
DS_NAMES = [
    "cancer",
    "glass",
    "magic",
    "spambase",
    "vertebral",
    "loans",
]

DS_PATHS = {
    "cancer": os.path.join(script_directory, "data/cancer/wdbc.data"),
    "glass": os.path.join(script_directory, "data/glass/glass.data"),
    "magic": os.path.join(script_directory, "data/magic/magic04.data"),
    "spambase": os.path.join(script_directory, "data/spambase/spambase.data"),
    "vertebral": os.path.join(script_directory, "data/vertebral/column_2C.dat"),
    "loans": os.path.join(script_directory, "data/loans/loans_continuous.csv"),
}

DS_DIMENSIONS = {
    "cancer": (568, 30),
    "glass": (162, 9),
    "magic": (19019, 10),
    "spambase": (4600, 57),
    "vertebral": (309, 6),
    "loans": (578, 4),
}


def load_data(dataset_name, preprocessing: str = "Normalize"):
    """
    Returns one of many possible anomaly detetion datasets based on the given `dataset_name`

    Parameters
    ----------
    dataset_name : the abbreviated name of the dataset to load, see README for all datset options
    preprocessing: if None do nothing, if Normalize normalize all features [0,1], if Scale Standardize features by
        removing the mean and scaling to unit variance

    Returns
    -------
    x : the dataset samples with shape (nsamples, nfeatures)
    y : the dataset labels with shape (nsamples,) containing 0 to represent a normal label and 1 for an anomaly label
    """
    if dataset_name == "cancer":
        x, y, colnames = util_load_cancer()
    elif dataset_name == "glass":
        x, y, colnames = util_load_glass()
    elif dataset_name == "magic":
        x, y, colnames = util_load_magic()
    elif dataset_name == "spambase":
        x, y, colnames = util_load_spambase()
    elif dataset_name == "vertebral":
        x, y, colnames = util_load_vertebral()
    elif dataset_name == "loans":
        x, y, colnames = util_load_loans()
    else:
        print("ERROR NO SUCH DATASET")
        exit(0)

    if preprocessing == "Normalize":
        if DO_VIZUALIZATION:
            save_data_dict(
                dataset_name,
                x,
                y,
                colnames,
                path=VIZ_DATA_PATH + "dataset_details.json",
                normalize=True,
            )
        # normalize x to [0, 1]
        min_value = np.min(x, axis=0)
        max_value = np.max(x, axis=0)
        x = (x - min_value) / (max_value - min_value)
    elif preprocessing == "Scale":
        float_transformer = StandardScaler()
        x = float_transformer.fit_transform(x)
    return x, y


def util_load_cancer():
    data = pd.read_csv(DS_PATHS["cancer"])
    colnames = list(data.columns)
    data = data.to_numpy()
    labels = data[:, 1:2].squeeze()  # lables of M=Malignant, B=Benign

    # label malignant as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "M"] = 1
    x = data[:, 2:].astype(float)
    return x, y, colnames


def util_load_glass():
    data = pd.read_csv(DS_PATHS["glass"])
    colnames = list(data.columns)
    data = data.to_numpy()
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
    return x, y, colnames


def util_load_magic():
    data = pd.read_csv(DS_PATHS["magic"])
    colnames = list(data.columns)
    data = data.to_numpy()
    # lables of g=gamma (signal), h=hadron (background)
    labels = data[:, -1:].squeeze()

    # label gamma (signal) as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "g"] = 1
    x = data[:, :-1].astype(float)
    return x, y, colnames


def util_load_spambase():
    path = DS_PATHS["spambase"]
    data = pd.read_csv(path)
    colnames = list(data.columns)
    data = data.to_numpy()
    x = data[:, :-1]
    y = data[:, -1:].squeeze().astype("int")
    return x, y, colnames


def util_load_vertebral():
    data = pd.read_csv(DS_PATHS["vertebral"], delim_whitespace=True)
    colnames = list(data.columns)
    data = data.to_numpy()
    # lables of g=gamma (signal), h=hadron (background)
    labels = data[:, -1:].squeeze()

    # label gamma (signal) as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "AB"] = 1
    x = data[:, :-1].astype(float)
    return x, y, colnames


def util_load_loans():
    data = pd.read_csv(DS_PATHS["loans"])
    colnames = list(data.columns)
    data = data.dropna().to_numpy()
    labels = data[:, -1:].squeeze()

    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "Y"] = 1
    x = data[:, :-1].astype(float)
    return x, y, colnames
