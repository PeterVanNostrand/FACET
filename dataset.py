from dataclasses import dataclass
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
import os
from pathlib import Path


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


@dataclass
class DataInfo:
    ncols: int
    is_normalized: bool
    col_scales: dict
    col_names: dict
    col_to_idx: dict = None

    def __post_init__(self):
        """
        Runs once after object creation
        """
        col_to_idx = dict()
        for i in range(self.ncols):
            col_to_idx["x{}".format(i)] = i
        self.col_to_idx = col_to_idx

    def unscale_points(self, x: np.ndarray):
        """
        Unscales the given points, x must be an array/list of shape (ninstances, ncols)
        """
        # if the data is not normalized, do nothing
        if not self.is_normalized:
            return x
        unscaled = x.copy()
        if len(unscaled.shape) == 1:  # if we only have one instance
            unscaled = unscaled.reshape(-1, unscaled.shape[0])
        # for each instace, unscale the data
        for i in range(unscaled.shape[0]):
            for col_id in range(self.ncols):
                min_val, max_val = self.col_scales[col_id]
                unscaled[i, col_id] = (
                    unscaled[i, col_id] * (max_val - min_val) + min_val
                )
        unscaled = unscaled.squeeze()
        return unscaled

    def scale_points(self, x: np.ndarray):
        """
        Unscales the given points, x must be an array/list of shape (ninstances, ncols)
        """
        # if the data is not normalized, do nothing
        if not self.is_normalized:
            return x
        scaled = x.copy()
        if len(scaled.shape) == 1:  # if we only have one instance
            scaled = scaled.reshape(-1, scaled.shape[0])
        # for each instace, scale the data
        for i in range(scaled.shape[0]):
            for col_id in range(self.ncols):
                min_val, max_val = self.col_scales[col_id]
                scaled[i, col_id] = (scaled[i, col_id] - min_val) / (max_val - min_val)
        scaled = scaled.squeeze()
        return scaled

    def scale_rects(self, rects: np.ndarray):
        """
        Scales the given rectangles, rects must be an array/list of shape (nrects, ncols, 2)
        """
        LOWER, UPPER = 0, 1
        scaled_rects = rects.copy()

        if len(scaled_rects.shape) == 2:
            scaled_rects = scaled_rects.reshape(-1, self.ncols, 2)

        for i in range(len(scaled_rects)):  # for each rectangle
            for col_id in range(self.ncols):  # for each column
                min_val, max_val = self.col_scales[col_id]
                for end in [LOWER, UPPER]:  # for the lower and upper bound
                    scaled_rects[i][col_id][end] = (
                        scaled_rects[i][col_id][end] - min_val
                    ) / (max_val - min_val)

        return scaled_rects

    def unscale_rects(self, rects: np.ndarray):
        """
        Unscales the given rectangles, rects must be an array/list of shape (nrects, ncols, 2)
        """
        LOWER, UPPER = 0, 1
        unscaled_rects = rects.copy()
        if len(unscaled_rects.shape) == 2:
            unscaled_rects = unscaled_rects.reshape(-1, self.ncols, 2)
        for i in range(len(unscaled_rects)):  # for each rectangle
            for col_id in range(self.ncols):  # for each column
                for end in [LOWER, UPPER]:  # for the lower and upper bound
                    min_val, max_val = self.col_scales[col_id]
                    unscaled_rects[i][col_id][end] = (
                        unscaled_rects[i][col_id][end] * (max_val - min_val) + min_val
                    )
        unscaled_rects = unscaled_rects.squeeze()
        return unscaled_rects

    def rect_to_dict(self, rect: np.ndarray) -> dict:
        """
        Takes a numpy array representing a rectangular counterfactual region and converts it to a dict
        """
        INFTY = 100000000000000
        LOWER, UPPER = 0, 1
        rect_dict = dict()
        rect[rect == -np.inf] = -INFTY
        rect[rect == np.inf] = INFTY
        for i in range(self.ncols):
            rect_dict["x{:d}".format(i)] = [rect[i, LOWER], rect[i, UPPER]]
        return rect_dict


    def dict_to_point(self, point_dict: dict) -> np.ndarray:
        """
        Takes a dictionary of {"xi" : <number>, ...} and converts it to a numpy array
        """
        point = np.zeros(shape=(self.ncols,))
        for col, val in point_dict.items():
            point[self.col_to_idx[col]] = val
        return point

    def point_to_dict(self, point: np.ndarray) -> dict:
        """
        Takes a a numpy array and converts it to a dictionary of the form {"xi" : <number>, ...}
        """
        point_dict = dict()
        for i in range(point.shape[0]):
            point_dict["x{:d}".format(i)] = point[i]
        return point_dict


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

    is_normalized = preprocessing == "Normalize"
    # save the dataset details
    data_directory = Path(DS_PATHS[dataset_name]).parent
    save_data_dict(
        dataset_name,
        x,
        y,
        colnames,
        path=str(data_directory) + "/dataset_details.json",
        normalize=is_normalized,
    )

    # normalize x to [0, 1] if requested
    if is_normalized:
        min_value = np.min(x, axis=0)
        max_value = np.max(x, axis=0)
        x = (x - min_value) / (max_value - min_value)
    return x, y, min_value, max_value


def get_json_paths(dataset_name: str) -> Tuple[str, str]:
    """
    Returns the paths the JSON files corresponding to the given dataset information

    Parameters
    ----------
    dataset_name : the short name of a dataset from DS_NAMES

    Returns
    -------
    json_paths : a tuple[ds_details_path, human_readable_path]
    """
    # check the dataset name is valid
    if dataset_name not in DS_NAMES:
        print("ERROR NO SUCH DATASET")
        exit(0)

    # get the paths of the JSON files
    data_directory = str(Path(DS_PATHS[dataset_name]).parent)
    ds_details_path = data_directory + "/dataset_details.json"
    human_readable_path = data_directory + "/human_readable.json"

    # return them as a tuple
    json_paths = [ds_details_path, human_readable_path]
    return json_paths


def save_data_dict(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    colnames: List[str],
    path: str,
    normalize: bool,
):
    """
    Saves creates a dictionary of dataset details and saves it to JSON file
    """
    # check if the directory exists, if not create it
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    min_value = np.min(x, axis=0)
    max_value = np.max(x, axis=0)
    dataset_details = {
        "dataset": dataset_name,
        "n_features": x.shape[1],
        "normalized": normalize,
        "target_name": colnames[-1],
        "feature_names": {},
        "min_values": {},
        "max_values": {},
        "std_dev": {},
    }
    std_devs = np.std(x, axis=0)
    for i in range(x.shape[1]):
        feature_id = "x{:d}".format(i)
        dataset_details["feature_names"][feature_id] = colnames[i]
        dataset_details["min_values"][feature_id] = min_value[i]
        dataset_details["max_values"][feature_id] = max_value[i]
        dataset_details["std_dev"][feature_id] = std_devs[i]
    with open(path, "w") as f:
        json.dump(dataset_details, f, indent=4)


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
