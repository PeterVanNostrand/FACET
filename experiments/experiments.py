import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import load_data
from manager import MethodManager
from utilities.metrics import average_distance, classification_metrics, percent_valid

TUNED_FACET_SD = {
    "cancer": 0.1,
    "glass": 0.005,
    "magic": 0.001,
    "spambase": 0.01,
    "vertebral": 0.05,
    "loans": 0.05,  # TODO tune on loans
}

# generated as the average minimum point to point distance
# see initial_radius_heuristic
AVG_NN_DIST = {
    "cancer": 0.3290,
    "glass": 0.1516,
    "magic": 0.07162,
    "spambase": 0.1061,
    "vertebral": 0.0992,
}
# avgerage dataset distance to nearest point of the oppisite class
AVG_CF_NN_DIST = {
    "cancer": 0.5850,
    "glass": 0.2710,
    "magic": 0.1153,
    "spambase": 0.2594,
    "vertebral": 0.1640,
}
# TUNED_FACET_RADII = AVG_NN_DIST

FACET_TUNED_M = {
    "cancer": 16,
    "glass": 16,
    "magic": 16,
    "spambase": 16,
    "vertebral": 16,
    "loans": 16,  # TODO tune on loans
}


MACE_DEFAULT_PARAMS = {"mace_maxtime": 300, "mace_epsilon": 1e-7, "mace_verbose": False}

OCEAN_DEFAULT_PARAMS = {"ocean_norm": 2, "ocean_ilf": True}

FACET_DEFAULT_PARAMS = {
    "facet_offset": 0.0001,
    "facet_nrects": 20_000,
    "facet_enumerate": "PointBased",
    "facet_sample": "Augment",
    "facet_sd": 0.01,
    "facet_intersect_order": "Axes",
    "facet_verbose": False,
    "facet_search": "BitVector",
    # "facet_search": "Linear",
    "rbv_initial_radius": 0.01,
    "rbv_radius_step": 0.01,
    "rbv_radius_growth": "Linear",
    "rbv_num_interval": 16,
}

RFOCSE_DEFAULT_PARAMS = {
    "rfoce_transform": False,
    "rfoce_offset": 0.0001,
    "rfoce_maxtime": None,
}

AFT_DEFAULT_PARAMS = {"aft_offset": 0.0001}

RF_DEFAULT_PARAMS = {
    "rf_ntrees": 100,
    "rf_maxdepth": None,
    "rf_hardvoting": False,
}

DEFAULT_PARAMS = {
    "RandomForest": RF_DEFAULT_PARAMS,
    "FACETIndex": FACET_DEFAULT_PARAMS,
    "MACE": MACE_DEFAULT_PARAMS,
    "RFOCSE": RFOCSE_DEFAULT_PARAMS,
    "AFT": AFT_DEFAULT_PARAMS,
    "OCEAN": OCEAN_DEFAULT_PARAMS,
}


def check_create_directory(dir_path="./results"):
    """
    Checks the the directory at `dir_path` exists, if it does not it creates all directories in the path
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # find the next availible run_id in the specified results directory
    max_run_id = 0
    dir_names = os.listdir(dir_path)
    for name in dir_names:
        x = re.match("run-(\d{3})", name)  # noqa: W605 (ignore linting from flake)
        if x is not None:
            found_run_id = int(x.group(1))
            if found_run_id > max_run_id:
                max_run_id = found_run_id
    run_id = max_run_id + 1

    # return the run_id and a path to that folder
    run_dir = "run-{:03d}".format(run_id)
    run_path = os.path.join(os.path.abspath(dir_path), run_dir)
    os.makedirs(run_path)
    return run_id, run_path


def execute_run(
    dataset_name: str,
    explainer: str,
    params: dict,
    output_path: str,
    iteration: int,
    test_size=0.2,
    n_explain: int = None,
    random_state: int = None,
    preprocessing: str = "Normalize",
    run_ext="",
    undesired_only=False,
):
    """
    dataset_name: the name of a valid dataset to load see datasets.py
    explainer: string name of a valid explainer class
    params: a dictionary of hyper-parameters for the RFModel and explainer to use
    output_path: directory to store run configuration and explanations to, ending with "/"
    iteration: id of the iteration, appended to config and explantion file names
    test_size: what portion of the dataset to reserve for testing
    n_explain: the number of samples to explain, if set to None the entire testing set is explained
    random_state: int value use to reproducibly create the same model and data boostraps
    preprocessing: how to process the dataset. Options are None, `Normalize` (to [0,1]), and `Scale` (u=0, sigma=c)
    """
    # set appropriate random seeds for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)

    # create the output directory
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # store this runs configuration
    config = {}
    config["explainer"] = explainer
    config["iteration"] = iteration
    config["dataset_name"] = dataset_name
    config["preprocessing"] = preprocessing
    config["test_size"] = test_size
    config["n_explain"] = n_explain
    config["output_path"] = output_path
    config["random_state"] = random_state
    config["params"] = params
    with open(
        output_path + "{}_{}_{}{:03d}_config.json".format(
            dataset_name, explainer.lower(), run_ext, iteration
        ),
        "w",
    ) as f:
        json_text = json.dumps(config, indent=4)
        f.write(json_text)

    # load and split the datset using random state for repeatability. Select samples to explain
    x, y, min_value, max_value = load_data(dataset_name, preprocessing=preprocessing)
    indices = np.arange(start=0, stop=x.shape[0])
    xtrain, xtest, ytrain, ytest, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=test_size, shuffle=True, random_state=random_state
    )
    if n_explain is not None:
        x_explain = xtest[:n_explain]
        # y_explain = ytest[:n_explain]
        ixd_explain = idx_test[:n_explain]
    else:
        x_explain = xtest
        # y_explain = ytest
        ixd_explain = idx_test
        n_explain = x_explain.shape[0]

    # create the manager which handles create the RF model and explainer
    manager = MethodManager(
        explainer=explainer, hyperparameters=params, random_state=random_state
    )

    # train ane evalute the random forest model
    manager.train(xtrain, ytrain)
    preds = manager.predict(xtest)
    accuracy, precision, recall, f1 = classification_metrics(
        preds, ytest, verbose=False
    )

    if undesired_only:
        idx_undesired = preds == 0
        xtest = preds[idx_undesired]
        preds = preds[idx_undesired]

    # prepare the explainer, handles any neccessary preprocessing
    prep_start = time.time()
    manager.explainer.prepare_dataset(x, y)
    manager.prepare(xtrain=xtrain, ytrain=ytrain)
    prep_end = time.time()
    prep_time = prep_end - prep_start

    # explain the samples using RF predictions (not ground truth)
    explain_preds = manager.predict(x_explain)
    explain_start = time.time()
    instances, explanations = manager.explain(x_explain, explain_preds)

    explain_end = time.time()
    explain_time = explain_end - explain_start
    sample_time = explain_time / n_explain

    # store the returned explantions
    col_names = []
    for i in range(x.shape[1]):
        col_names.append("x{}".format(i))
    expl_df = pd.DataFrame(instances, columns=col_names)
    # also store the index of the explained sample in the dataset
    expl_df.insert(0, "x_idx", ixd_explain)
    explanation_path = output_path + "{}_{}_{}{:03d}_explns.csv".format(
        dataset_name, explainer.lower(), run_ext, iteration
    )
    expl_df.to_csv(explanation_path, index=False)

    # evalute the quality of the explanations
    per_valid = percent_valid(instances)
    avg_dist = average_distance(
        x_explain, instances, distance_metric="Euclidean"
    )  # L2 Norm Euclidean
    avg_manhattan = average_distance(
        x_explain, instances, distance_metric="Manhattan"
    )  # L1 Norm Manhattan
    avg_length = average_distance(
        x_explain, instances, distance_metric="FeaturesChanged"
    )  # L0 Norm Sparsity

    # store and return the top level results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_valid": per_valid,
        "avg_dist": avg_dist,
        "avg_manhattan": avg_manhattan,
        "avg_dist": avg_dist,
        "avg_length": avg_length,
        "prep_time": prep_time,
        "explain_time": explain_time,
        "sample_time": sample_time,
        "n_explain": n_explain,
    }

    json_path = output_path + "{}_{}_{}{:03d}_result.json".format(dataset_name, explainer.lower(), run_ext, iteration)

    with open(json_path, "w") as f:
        json_text = json.dumps(results, indent=4)
        f.write(json_text)

    return results
