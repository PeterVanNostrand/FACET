import numpy as np
import os
import cProfile
import time
import random
import math
import json
import re
from numpy.core.fromnumeric import var

from manager import MethodManager
import matplotlib.pyplot as plt
from utilities.metrics import percent_valid
from utilities.metrics import classification_metrics
from utilities.metrics import average_distance
from utilities.tree_tools import compute_jaccard
from sklearn.model_selection import train_test_split
from dataset import load_data
from dataset import DS_NAMES
from experiments import execute_run
from vary_nrects import vary_nrects
from vary_ntrees import vary_ntrees
from vary_sigma import vary_sigma


def check_create_directory(dir_path="./results/"):
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
    run_dir = "run-{:03d}/".format(run_id)
    run_path = os.path.join(os.path.abspath(dir_path), run_dir)
    os.makedirs(run_path)
    return run_id, run_path


def simple_run(dataset_name):
    # x, y = load_data(dataset_name, preprocessing=None)
    # xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

    # Euclidean, FeaturesChanged
    run_id, run_path = check_create_directory("./results/simple-run/")

    rf_params = {
        "rf_ntrees": 10,
        "rf_maxdepth": 5,
        "rf_hardvoting": False,  # note OCEAN and FACETIndex use soft and hard requirment
    }
    facet_params = {
        "facet_offset": 0.0001,
        "facet_nrects": 10_000,
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
        "mace_epsilon": 0.1
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

    explainer = "MACE"
    iteration = 0
    preprocessing = "Normalize"

    print("Run ID: {}".format(run_id))
    print("explainer: " + explainer)
    print("dataset: " + dataset_name)
    print("config:")
    print(json.dumps(params, indent=4))

    results = execute_run(
        dataset_name=dataset_name,
        explainer=explainer,
        params=params,
        output_path=run_path,
        iteration=iteration,
        test_size=0.2,
        n_explain=3,
        random_state=1,
        preprocessing=preprocessing
    )
    print("results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    RAND_SEED = 0
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    all_ds = ["cancer", "glass", "magic", "spambase", "vertebral"]
    all_explaiers = ["FACETIndex", "OCEAN", "RFOCSE", "AFT", "MACE"]
    notMACE = ["FACETIndex", "OCEAN", "RFOCSE", "AFT"]
    # index_test(ds_names=["vertebral"], exp_var="facet_nrects", exp_vals=[1000, 5000, 10000],
    #            num_iters=1, eval_samples=20, test_size=0.2, seed=RAND_SEED)

    # index_test(ds_names=run_ds, exp_var="facet_intersect_order", exp_vals=["Axes", "Size", "Ensemble"],
    #            num_iters=1, eval_samples=20, test_size=0.2, seed=RAND_SEED)
    # index_test()
    # run_ds.remove("spambase")
    # compare_methods(["vertebral"], num_iters=1, explainers=["MACE"], eval_samples=20, seed=RAND_SEED)
    # vary_ntrees(["vertebral"], explainer="FACETIndex", ntrees=list(range(5, 105, 5)), num_iters=5, seed=SEED)
    # simple_run("vertebral")

    # vary_ntrees(ds_names=all_ds, explainers=["FACETIndex", "OCEAN"], ntrees=[10, 50, 100, 150, 200], iterations=[0])
    # nrectangles = [1_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 100_000]
    # nrectangles = [70_000, 90_000]
    # vary_nrects(ds_names=all_ds, nrects=nrectangles, iterations=list(range(10)))
    # vary_nrects(all_ds, nrects=nrectangles, iterations=[1, 2, 3, 4])
    vary_sigma(ds_names=all_ds, sigmas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25], iterations=[0])
