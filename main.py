import numpy as np
import os
import random
import json
import re
from numpy.core.fromnumeric import var
import argparse

from manager import MethodManager
from utilities.metrics import percent_valid
from utilities.metrics import classification_metrics
from utilities.metrics import average_distance
from utilities.tree_tools import compute_jaccard
from sklearn.model_selection import train_test_split
from dataset import load_data
from dataset import DS_NAMES
from experiments import execute_run, DEFAULT_PARAMS
from vary_nrects import vary_nrects
from vary_ntrees import vary_ntrees
from vary_sigma import vary_sigma
from vary_eps import vary_eps
from vary_enum import vary_enum
from compare_methods import compare_methods


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


def simple_run(ds_name="vertebral", explainer="FACETIndex", random_state=0):
    # Euclidean, FeaturesChanged
    run_id, run_path = check_create_directory("./results/simple-run/")

    ntrees = 10
    max_depth = 5
    params = DEFAULT_PARAMS
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth
    params["OCEAN"]["ocean_ilf"] = False

    preprocessing = "Normalize"
    n_explain = 20

    print("Run ID: {}".format(run_id))
    print("explainer: " + explainer)
    print("dataset: " + ds_name)
    print("config:")
    print(json.dumps(params, indent=4))

    results = execute_run(
        dataset_name=ds_name,
        explainer=explainer,
        params=params,
        output_path=run_path,
        iteration=random_state,
        test_size=0.2,
        n_explain=n_explain,
        random_state=random_state,
        preprocessing=preprocessing
    )
    print("results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":

    all_ds = ["cancer", "glass", "magic", "spambase", "vertebral"]
    all_explaiers = ["FACETIndex", "OCEAN", "RFOCSE", "AFT", "MACE"]

    parser = argparse.ArgumentParser(description='Run FACET Experiments')
    # epsilon - MACE experiment for step size in their exploration
    # sigma - FACET index evaluation (how much do we move our augmented data in region enumeration)
    # compare - 7.3 compare methods on fixed ensemble
    parser.add_argument("--expr", choices=["simple", "ntrees", "nrects",
                        "eps", "sigma", "enum", "compare"], default="simple")
    parser.add_argument("--ds", type=str, nargs="+", default=["vertebral"])
    parser.add_argument("--method", type=str, nargs="+", choices=all_explaiers, default=["FACETIndex"])
    parser.add_argument("--values", type=float, nargs="+", default=None)
    parser.add_argument("--it", type=int, nargs="+", default=[0])
    parser.add_argument("--fmod", type=str, default=None)

    args = parser.parse_args()

    print(args)

    # Do a single quick run with one explaienr and one dataset
    if args.expr == "simple":
        simple_run(ds_name=args.ds[0], explainer=args.method[0], random_state=args.it[0])
    # Vary the number of trees and compare explaienrs
    elif args.expr == "ntrees":
        if args.values is not None:
            ntrees = [int(_) for _ in args.values]
            vary_ntrees(ds_names=args.ds, explainers=args.method, ntrees=ntrees, iterations=args.it, fmod=args.fmod)
        else:
            vary_ntrees(ds_names=args.ds, explainers=args.method, iterations=args.it, fmod=args.fmod)

    # Vary the number of hyperrectangles for FACETIndex
    elif args.expr == "nrects":
        if args.values is not None:
            nrects = [int(_) for _ in args.values]
            vary_nrects(ds_names=args.ds, nrects=nrects, iterations=args.it, fmod=args.fmod)
        else:
            vary_nrects(ds_names=args.ds, iterations=args.it, fmod=args.fmod)

    # Vary the epsilon value for MACE
    elif args.expr == "eps":
        if args.values is not None:
            vary_eps(ds_names=args.ds, epsilons=args.values, iterations=args.it, fmod=args.fmod)
        else:
            vary_eps(ds_names=args.ds, iterations=args.it, fmod=args.fmod)

    # Vary the standard deviation of HR enumeration for FACETIndex
    elif args.expr == "sigma":
        if args.values is not None:
            vary_sigma(ds_names=args.ds, sigmas=args.values, iterations=args.it, fmod=args.fmod)
        else:
            vary_sigma(ds_names=args.ds, iterations=args.it, fmod=args.fmod)

    elif args.expr == "enum":
        vary_enum(ds_names=args.ds, iterations=args.it, fmod=args.fmod)

    elif args.expr == "compare":
        compare_methods(ds_names=args.ds, explainers=args.method, iterations=args.it, fmod=args.fmod)
