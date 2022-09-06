import numpy as np
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

# from experiments_old import *
import cProfile
import time
import random
import math
import json
from experiments import execute_run, vary_ntrees


def simple_run(dataset_name):
    # x, y = load_data(dataset_name, preprocessing=None)
    # xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

    # Euclidean, FeaturesChanged
    rf_params = {
        "rf_maxdepth": 2,
        "rf_ntrees": 3,
        "rf_hardvoting": False,  # note OCEAN and FACETIndex use soft and hard requirment
    }
    facet_params = {
        "facet_offset": 0.0001,
        "facet_nrects": 10000,
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

    explainer = "MACE"
    iteration = 0
    preprocessing = "Normalize"
    results = execute_run(
        dataset_name=dataset_name,
        explainer=explainer,
        params=params,
        output_path="results/test-run/",
        iteration=iteration,
        test_size=0.2,
        n_explain=3,
        random_state=1,
        preprocessing=preprocessing
    )
    print("explainer: " + explainer)
    print("dataset: " + dataset_name)
    json_text = json.dumps(results, indent=4)
    print(json_text)


if __name__ == "__main__":
    RAND_SEED = 0
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    run_ds = DS_NAMES.copy()
    # index_test(ds_names=["vertebral"], exp_var="facet_nrects", exp_vals=[1000, 5000, 10000],
    #            num_iters=1, eval_samples=20, test_size=0.2, seed=RAND_SEED)

    # index_test(ds_names=run_ds, exp_var="facet_intersect_order", exp_vals=["Axes", "Size", "Ensemble"],
    #            num_iters=1, eval_samples=20, test_size=0.2, seed=RAND_SEED)
    # index_test()
    # run_ds.remove("spambase")
    # compare_methods(["vertebral"], num_iters=1, explainers=["MACE"], eval_samples=20, seed=RAND_SEED)
    # vary_ntrees(run_ds, explainer="FACETIndex", ntrees=list(range(5, 105, 5)), num_iters=5, seed=SEED)
    # simple_run("magic")
    vary_ntrees(ds_names=["vertebral"], explainers=["FACETIndex"], ntrees=[10], num_iters=1)
    # bb_ntrees(run_ds, ntrees=[25], depths=[3], num_iters=1, eval_samples=5)
    # hard_vs_soft(run_ds, num_iters=10)
    # bb_ordering(run_ds, orderings=["PriorityQueue", "Stack", "ModifiedPriorityQueue"], num_iters=1,
    # test_size=0.2, ntrees=15, max_depth=3, eval_samples=5)
    # 100, 1000, 5000, 10000, 20000, 30000, 40000, 50000
    # vary_nrects(run_ds, nrects=[20000, 40000, 60000, 80000, 100000, 150000,
    # 200000, 250000], num_iters=1, eval_samples=20, seed=RAND_SEED)
