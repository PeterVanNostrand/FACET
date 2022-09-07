from distutils.command import build
from genericpath import isfile
from pydoc import ispath
from typing_extensions import runtime
from unittest import result
from sklearn.model_selection import train_test_split
import os
import re

import random
import pandas as pd
import numpy as np
import time
from tqdm.auto import tqdm
import json

from manager import MethodManager
from dataset import load_data
from dataset import DS_DIMENSIONS
from utilities.metrics import classification_metrics, percent_valid, average_distance


def check_create_directory(dir_path="./results"):
    '''
    Checks the the directory at `dir_path` exists, if it does not it creates all directories in the path
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # find the next availible run_id in the specified results directory
    max_run_id = 0
    dir_names = os.listdir(dir_path)
    for name in dir_names:
        x = re.match("run-(\d{3})", name)
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


def execute_run(dataset_name: str, explainer: str, params: dict, output_path: str, iteration: int, test_size=0.2, n_explain: int = None, random_state: int = None, preprocessing: str = "Normalize", run_ext=""):
    '''
    dataset_name: the name of a valid dataset to load see datasets.py
    explainer: string name of a valid explainer class
    params: a dictionary of hyper-parameters for the RFModel and explainer to use
    output_path: directory to store run configuration and explanations to, ending with "/"
    iteration: id of the iteration, appended to config and explantion file names
    test_size: what portion of the dataset to reserve for testing
    n_explain: the number of samples to explain, if set to None the entire testing set is explained
    random_state: int value use to reproducibly create the same model and data boostraps
    preprocessing: how to process the dataset. Options are None, `Normalize` (to [0,1]), and `Scale` (u=0, sigma=c)
    '''
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
    with open(output_path + "{}_{}_{}{:03d}_config.json".format(dataset_name, explainer.lower(), run_ext, iteration), "w") as f:
        json_text = json.dumps(config, indent=4)
        f.write(json_text)

    # load and split the datset using random state for repeatability. Select samples to explain
    x, y = load_data(dataset_name, preprocessing=preprocessing)
    indices = np.arange(start=0, stop=x.shape[0])
    xtrain, xtest, ytrain, ytest, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=test_size, shuffle=True, random_state=random_state)
    if n_explain is not None:
        x_explain = xtest[:n_explain]
        y_explain = ytest[:n_explain]
        ixd_explain = idx_test[:n_explain]
    else:
        x_explain = xtest
        y_explain = ytest
        ixd_explain = idx_test
        n_explain = x_explain.shape[0]

    # create the manager which handles create the RF model and explainer
    manager = MethodManager(explainer=explainer, hyperparameters=params, random_state=random_state)

    # train ane evalute the random forest model
    manager.train(xtrain, ytrain)
    preds = manager.predict(xtest)
    accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)

    # prepare the explainer, handles any neccessary preprocessing
    prep_start = time.time()
    manager.explainer.prepare_dataset(x, y)
    manager.prepare(data=xtrain)
    prep_end = time.time()
    prep_time = prep_end-prep_start

    # explain the samples using RF predictions (not ground truth)
    explain_preds = manager.predict(x_explain)
    explain_start = time.time()
    explanations: np.ndarray = manager.explain(x_explain, explain_preds)

    explain_end = time.time()
    explain_time = explain_end - explain_start
    sample_time = explain_time / n_explain

    # store the returned explantions
    col_names = []
    for i in range(x.shape[1]):
        col_names.append("x{}".format(i))
    expl_df = pd.DataFrame(explanations, columns=col_names)
    # also store the index of the explained sample in the dataset
    expl_df.insert(0, "x_idx", ixd_explain)
    explanation_path = output_path + \
        "{}_{}_{}{:03d}_explns.csv".format(dataset_name, explainer.lower(), run_ext, iteration)
    expl_df.to_csv(explanation_path, index=False)

    # evalute the quality of the explanations
    per_valid = percent_valid(explanations)
    avg_dist = average_distance(x_explain, explanations, distance_metric="Euclidean")
    avg_length = average_distance(x_explain, explanations, distance_metric="FeaturesChanged")

    # store and return the top level results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_valid": per_valid,
        "avg_dist": avg_dist,
        "avg_length": avg_length,
        "prep_time": prep_time,
        "explain_time": explain_time,
        "sample_time": sample_time,
        "n_explain": n_explain,
    }

    with open(output_path + "{}_{}_{}{:03d}_result.json".format(dataset_name, explainer.lower(), run_ext, iteration), "w") as f:
        json_text = json.dumps(results, indent=4)
        f.write(json_text)

    return results


def vary_ntrees(ds_names, explainers=["FACETIndex", "OCEAN", "RFOCSE", "AFT", "MACE"], ntrees=[5, 10, 15],
                iterations=[0, 1, 2, 3, 4]):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    print("Varying number of trees:")
    print("\tds_names:", ds_names)
    print("\texplainers:", explainers)
    print("\tntrees:", ntrees)
    print("\titerations:", iterations)

    experiment_path = "./results/vary-ntrees/"
    max_depth = 5
    rf_params = {
        "rf_maxdepth": max_depth,
        "rf_ntrees": -1,
        "rf_hardvoting": False,  # note OCEAN and FACETIndex use soft and hard requirment
    }
    facet_params = {
        "facet_offset": 0.001,
        "facet_nrects": 20000,
        "facet_sample": "Augment",
        "facet_enumerate": "PointBased",
        "facet_verbose": False,
        "facet_sd": 0.3,
        "facet_search": "BitVector",
        "rbv_initial_radius": 0.01,
        "rbv_radius_growth": "Linear",
        "rbv_num_interval": 4
    }
    rfocse_params = {
        "rfoce_transform": False,
        "rfoce_offset": 0.0001
    }
    aft_params = {
        "aft_offset": 0.0001
    }
    mace_params = {
        "mace_maxtime": 300,
        "mace_epsilon": 0.01
    }
    ocean_params = {
        "ocean_norm": 2
    }
    params = {
        "RandomForest": rf_params,
        "FACETIndex": facet_params,
        "MACE": mace_params,
        "RFOCSE": rfocse_params,
        "AFT": aft_params,
        "OCEAN": ocean_params,
    }

    total_runs = len(ds_names) * len(explainers) * len(ntrees) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for ds in ds_names:
        for expl in explainers:
            for ntree in ntrees:
                for iter in iterations:
                    # set the number of trees
                    params["RandomForest"]["rf_ntrees"] = ntree
                    # FACET uses hardvoting
                    if expl == "FACETIndex":
                        params["RandomForest"]["rf_hardvoting"] = True
                    else:
                        params["RandomForest"]["rf_hardvoting"] = False
                    run_result = execute_run(
                        dataset_name=ds,
                        explainer=expl,
                        params=params,
                        output_path=experiment_path,
                        iteration=iter,
                        test_size=0.2,
                        n_explain=20,
                        random_state=1,
                        preprocessing="Normalize",
                        run_ext="t{:03d}_".format(ntree)
                    )
                    df_item = {
                        "dataset": ds,
                        "explainer": expl,
                        "n_trees": ntree,
                        "iteration": iter,
                        "max_depth": max_depth,
                        **run_result
                    }
                    experiment_results = pd.DataFrame([df_item])
                    if not os.path.exists(experiment_path + "vary_ntrees.csv"):
                        experiment_results.to_csv(experiment_path + "vary_ntrees.csv", index=False)
                    else:
                        experiment_results.to_csv(experiment_path + "vary_ntrees.csv",
                                                  index=False, mode="a", header=False,)

                    progress_bar.update()
    progress_bar.close()
    print("Finished varying number of trees")
