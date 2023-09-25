import json
import numpy as np
from typing import List
import os
from functools import lru_cache


LOWER = 0
UPPER = 1

instance_directory_created = False
all_explanation_paths = []


@lru_cache(maxsize=1)
def check_create_directory(path):
    '''
    Creates all directories along the given path. Function is cached such that if it called multiple times with the same arguement it only runs once and returns the previous result. This prevents performing a redundanat and slow I/O operation on every call
    '''
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def save_data_dict(dataset_name: str, x: np.ndarray, y: np.ndarray, colnames: List[str], path: str, normalize: bool):
    '''
    Saves creates a dictionary of dataset details and saves it to JSON file
    '''
    check_create_directory(path)
    min_value = np.min(x, axis=0)
    max_value = np.max(x, axis=0)
    dataset_details = {
        "dataset": dataset_name,
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


def save_instance_JSON(x: np.ndarray, path: str):
    '''
    Saves an instance to a JSON object

    `x` : a numpy array of shape (nfeatures,) containing one instance
    `path` : the path to save the JSON file
    '''
    check_create_directory(path)
    # convert instance to dict and save as JSON
    instance_dict = {}
    for i in range(x.shape[0]):
        instance_dict["x{:d}".format(i)] = x[i]
    with open(path, "w") as f:
        json.dump(instance_dict, f, indent=4)


def save_instance_region_JSON(x: np.ndarray, rect: np.ndarray, path: str):
    '''
    Saves an instance and its region explantion to a JSON object and stores the save location of each file

    `x` : a numpy array of shape (nfeatures,) containing one instance
    `rect` : a numpy array of shape ()
    `path` : the path to save the JSON file
    '''
    check_create_directory(path)
    all_explanation_paths.append("/" + os.path.relpath(path).replace("\\", "/"))
    # convert instance to dict and save as JSON
    expl_dict = {
        "instance": {},
        "region": {}
    }
    # JSON does not support +/- infinity, replace with +/-100 trillion
    rect[rect == -np.inf] = -100000000000000
    rect[rect == np.inf] = 100000000000000
    for i in range(x.shape[0]):
        expl_dict["instance"]["x{:d}".format(i)] = x[i]
        expl_dict["region"]["x{:d}".format(i)] = [rect[i, LOWER], rect[i, UPPER]]
    with open(path, "w") as f:
        json.dump(expl_dict, f, indent=4)


def save_JSON_paths(path: str):
    '''
    Writes the save paths of all explantion files to a JSON file
    '''
    check_create_directory(path)
    with open(path, "w") as f:
        json.dump(all_explanation_paths, f, indent=4)
