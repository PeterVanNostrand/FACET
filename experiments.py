from distutils.command import build
from sklearn.model_selection import train_test_split
import os
import re

import pandas as pd
import numpy as np
import time
from tqdm.auto import tqdm
import json

from heead import HEEAD
from dataset import load_data
from dataset import DS_DIMENSIONS

from utilities.metrics import average_distance, coverage
from utilities.metrics import classification_metrics
from utilities.tree_tools import compute_jaccard


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


def execute_run(model: HEEAD, xtrain, xtest, ytrain, ytest):
    '''
    A general method for testing the given model with the provided data

    Parameters
    ----------
    model  : the model of type HEEAD to be evaluated
    xtrain : the training instances
    xtest  : the testing instances
    ytrain : the training labels
    ytest  : the testing labels

    Returns
    -------
    run_perf : a dictionary of important performance metrics for the model on the given data
    '''
    model.prepare()
    model.train(xtrain, ytrain)
    preds = model.predict(xtest)

    # generate the explanations
    explanations = model.explain(xtest, preds)

    # measure model performance
    accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)
    coverage_ratio = coverage(explanations)
    mean_dist = average_distance(xtest, explanations, distance_metric="Euclidean")
    mean_length = average_distance(xtest, explanations, distance_metric="FeaturesChanged")

    # save the performance
    run_perf = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage_ratio": coverage_ratio,
        "mean_distance": mean_dist,
        "mean_length": mean_length
    }

    return run_perf


def vary_difference():
    '''
    Experiment to observe the effect of the difference value using in decision tree explanation
    '''
    # output directory
    run_id, run_path = check_create_directory("./results/vary-difference/")

    # run configuration
    num_iters = 10
    differences = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    dets = ["RandomForest"]
    agg = "LogisticRegression"
    expl = "AFT"
    params = {
        "rf_distance": "Euclidean",
        "rf_k": 1,
        "rf_ntrees": 20
    }

    # save the run information
    with open(run_path + "/" + "config.txt", 'a') as f:
        f.write("varying the difference used during explanation\n\n")
        f.write("iterations: {:d}\n".format(num_iters))
        f.write("differences: ")
        for diff in differences:
            f.write(str(diff) + ", ")
        f.write("\n\n")
        f.write("detectors: ")
        for d in dets:
            f.write(d + ", ")
        f.write("\n")
        f.write("aggregator: " + agg + "\n")
        f.write("explainer: " + expl + "\n\n")
        f.write("hyperparameters{\n")
        for k in params.keys():
            f.write("\t" + k + ": " + str(params[k]) + "\n")
        f.write("}\n")

    for ds_name in ["thyroid", "cardio", "wbc", "musk"]:
        runs_complete = 0
        # Load the dataset
        x, y = load_data(ds_name, normalize=True)

        # dataframe to store results of all experimental runs
        results = pd.DataFrame(columns=["difference", "accuracy", "precision",
                                        "recall", "f1", "coverage_ratio", "mean_distance"])

        for diff in differences:
            params["rf_difference"] = diff
            for i in range(num_iters):
                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

                # Create, train, and predict with the model
                model = HEEAD(detectors=dets, aggregator=agg, explainer=expl, hyperparameters=params)
                run_perf = execute_run(model, xtrain, xtest, ytrain, ytest)

                # store the results
                diff_val = {"difference": diff}
                run_result = {**diff_val, **run_perf}
                results = results.append(run_result, ignore_index=True)

                # log progress
                runs_complete += 1
                print("\truns complete:", runs_complete)
        # save the results
        results.to_csv(run_path + "/" + ds_name + ".csv")
        print("finished", ds_name)


def vary_k():
    '''
    Experiment to observe the effect of the difference value using in decision tree explanation
    '''
    # output directory
    run_id, run_path = check_create_directory("./results/vary-k/")

    # run configuration
    kmin = 1
    kmax = 15
    num_iters = 10

    dets = ["RandomForest"]
    agg = "LogisticRegression"
    expl = "AFT"
    params = {
        "rf_difference": 0.01,
        "rf_distance": "Euclidean",
        "rf_ntrees": 20
    }

    # save the run information
    with open(run_path + "/" + "config.txt", 'a') as f:
        f.write("varying the number of candidate examples k\n\n")
        f.write("iterations: {:d}\n".format(num_iters))
        f.write("kmin: {:d}\n".format(kmin))
        f.write("kmax: {:d}\n\n".format(kmax))
        f.write("detectors: ")
        for d in dets:
            f.write(d + ", ")
        f.write("\n")
        f.write("aggregator: " + agg + "\n")
        f.write("explainer: " + expl + "\n\n")
        f.write("hyperparameters{\n")
        for k in params.keys():
            f.write("\t" + k + ": " + str(params[k]) + "\n")
        f.write("}\n")

    for ds_name in ["thyroid", "cardio", "wbc", "musk"]:
        runs_complete = 0
        # Load the dataset
        x, y = load_data(ds_name, normalize=True)

        # dataframe to store results of all experimental runs
        results = pd.DataFrame(columns=["k", "accuracy", "precision",
                                        "recall", "f1", "coverage_ratio", "mean_distance"])

        for k in range(kmin, kmax + 1):
            params["rf_k"] = k
            params["explainer_k"] = k

            for i in range(num_iters):
                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

                # Create, train, and predict with the model
                model = HEEAD(detectors=dets, aggregator=agg, explainer=expl, hyperparameters=params)
                run_perf = execute_run(model, xtrain, xtest, ytrain, ytest)

                # store results
                diff_val = {"k": k}
                run_result = {**diff_val, **run_perf}
                results = results.append(run_result, ignore_index=True)

                # log progress
                runs_complete += 1
                print("\truns complete:", runs_complete)
        # save the results
        results.to_csv(run_path + "/" + ds_name + ".csv")
        print("finished", ds_name)


def vary_dim(ds_names, explainer="AFT", distance="Euclidean"):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    # output directory
    run_id, run_path = check_create_directory("./results/vary-dim/")

    # run configuration
    num_iters = 5
    dets = ["RandomForest"]
    agg = "LogisticRegression"
    expl = explainer
    params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": 20,
        "expl_distance": distance
    }

    # save the run information
    with open(run_path + "/" + "config.txt", 'a') as f:
        f.write("varying the number of features\n\n")
        f.write("iterations: {:d}\n\n".format(num_iters))
        f.write("detectors: ")
        for d in dets:
            f.write(d + ", ")
        f.write("\n")
        f.write("aggregator: " + agg + "\n")
        f.write("explainer: " + expl + "\n\n")
        f.write("hyperparameters{\n")
        for k in params.keys():
            f.write("\t" + k + ": " + str(params[k]) + "\n")
        f.write("}\n")

    # compute the total number of runs for this experiment
    total_runs = 0
    for ds in ds_names:
        total_runs += (DS_DIMENSIONS[ds][1] - 1) * num_iters

    # perform the experiment
    print("Varying Dim", ds_names)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0)
    for ds in ds_names:
        # dataframe to store results of each datasets runs
        results = pd.DataFrame(columns=["n_features", "accuracy", "precision",
                               "recall", "f1", "coverage_ratio", "mean_distance", "avg_nnodes", "avg_nleaves", "avg_depth", "q", "jaccard"])
        progress_bar_ds = tqdm(total=(DS_DIMENSIONS[ds][1] - 1) * num_iters, desc=ds, leave=False)

        x_orig, y_orig = load_data(ds, normalize=True)
        for i in range(num_iters):
            # Load the dataset
            x = x_orig.copy()
            y = y_orig.copy()

            # randomly shuffle the order of the columns
            np.random.shuffle(np.transpose(x))

            # try first 1 feature, first 2 feature, ... , first n features
            for n in range(2, x.shape[1] + 1):
                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)
                # Create, train, and predict with the model
                model = HEEAD(detectors=dets, aggregator=agg, explainer=expl, hyperparameters=params)
                run_perf = execute_run(model, xtrain[:, :n], xtest[:, :n], ytrain, ytest)

                # get metrics of random forest trees
                avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()

                # store results
                ind_var = {"n_features": n}
                tree_stats = {
                    "avg_nnodes": avg_nnodes,
                    "avg_nleaves": avg_nleaves,
                    "avg_depth": avg_depth
                }
                Q, qs = model.detectors[0].compute_qs(xtest[:, :n], ytest)
                J, jaccards = compute_jaccard(model.detectors[0])
                diversity_info = {
                    "q": Q,
                    "jaccard": J
                }
                run_result = {**ind_var, **run_perf, **tree_stats, **diversity_info}
                results = results.append(run_result, ignore_index=True)

                # log progress
                progress_bar.update()
                progress_bar_ds.update()
        # save the results
        results.to_csv(run_path + "/" + ds + ".csv")
        progress_bar_ds.close()
    progress_bar.close()
    print("Finished varying dimension")


def vary_ntrees(ds_names, explainer="FACETIndex", ntrees=[5, 10, 15], distance="Euclidean", num_iters=5, eval_samples=20, test_size=0.2):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    # output directory
    run_id, run_path = check_create_directory("./results/vary-ntrees/")

    # run configuration
    min_trees = 1
    max_trees = 100
    num_iters = 1
    dets = ["RandomForest"]
    agg = "NoAggregator"
    expl = explainer
    params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": 20,
        "rf_threads": 1,
        "rf_maxdepth": 3,
        "expl_greedy": False,
        "expl_distance": distance,
        "ocean_norm": 2,
        "mace_maxtime": 300,
        "num_iters": num_iters,
        "eval_samples": eval_samples,
        "test_size": test_size,
        "facet_graphtype": "disjoint",
        "facet_offset": 0.001,
        "facet_mode": "exhaustive",
        "verbose": True,
        "rf_hardvoting": True,  # TODO consider OCEAN vs FACETIndex soft vs hard requirment
    }

    # save the run information
    with open(run_path + "/" + "config.txt", 'a') as f:
        f.write("comparing explanation methods\n\n")
        f.write("iterations: {:d}\n\n".format(num_iters))
        f.write("detectors: ")
        for d in dets:
            f.write(d + ", ")
        f.write("\n")
        f.write("aggregator: " + agg + "\n")
        f.write("explainer:" + explainer + "\n")
        f.write("\n")
        f.write("hyperparameters{\n")
        for k in params.keys():
            f.write("\t" + k + ": " + str(params[k]) + "\n")
        f.write("}\n")
        f.write("ntrees: ")
        for n in ntrees:
            f.write(str(n) + ", ")

    print("Varying ntrees")
    print("\tDatasets:", ds_names)
    print("\ntrees:", ntrees)

    # compute the total number of runs for this experiment
    total_runs = len(ds_names) * len(ntrees) * num_iters
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=True)

    for ds in ds_names:
        progress_bar_ds = tqdm(total=len(ntrees) * num_iters, desc=ds, leave=False)

        x, y = load_data(ds, normalize=True)
        # dataframe to store results of each datasets runs
        results = pd.DataFrame(columns=["n_trees", "explainer", "n_samples", "n_samples_explained", "n_features", "accuracy", "precision", "recall", "f1", "avg_nnodes", "avg_nleaves",
                               "avg_depth", "q", "jaccard", "coverage_ratio", "mean_distance", "mean_length", "init_time", "runtime", "clique_size", "grown_clique_size", "ext_min", "ext_avg", "ext_max"])

        for n in ntrees:
            params["rf_ntrees"] = n
            for i in range(num_iters):
                # random split the data
                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, shuffle=True)
                n_samples = DS_DIMENSIONS[ds][0]
                n_features = DS_DIMENSIONS[ds][1]

                if eval_samples is not None:
                    xtest = xtest[:eval_samples]
                    ytest = ytest[:eval_samples]
                    n_samples = eval_samples

                # Create and train the model
                model = HEEAD(detectors=dets, aggregator=agg, hyperparameters=params)
                model.set_explainer(expl, hyperparameters=params)
                model.train(xtrain, ytrain)
                preds = model.predict(xtest)

                # compute forest metrics
                accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)
                avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()
                Q, qs = model.detectors[0].compute_qs(xtest, ytest)
                J, jaccards = compute_jaccard(model.detectors[0])

                # explain instances
                start_build = time.time()
                model.prepare(data=xtrain)
                end_build = time.time()
                start = time.time()
                explanations = model.explain(xtest, preds)
                end = time.time()
                runtime = end-start  # wall time in seconds
                init_time = end_build - start_build

                # collect optional statistics
                if expl == "FACETTrees" or expl == "FACETPaths":
                    clique_size = model.explainer.get_clique_size()
                    grown_clique_size = -1
                elif expl == "FACETGrow":
                    clique_size, grown_clique_size = model.explainer.get_clique_size()
                elif expl == "FACETBranchBound":
                    clique_size = -1
                    grown_clique_size = -1
                    ext_min = model.explainer.ext_min
                    ext_avg = model.explainer.ext_avg
                    ext_max = model.explainer.ext_max
                else:
                    clique_size = -1
                    grown_clique_size = -1
                    ext_min = -1
                    ext_avg = -1
                    ext_max = -1

                # Compute explanation metrics
                coverage_ratio = coverage(explanations)
                mean_dist = average_distance(xtest, explanations, distance_metric="Euclidean")
                mean_length = average_distance(xtest, explanations, distance_metric="FeaturesChanged")

                # Save results
                # save the performance
                run_result = {
                    "n_trees": n,
                    "explainer": expl,
                    "n_samples": n_samples,
                    "n_samples_explained": xtest.shape[0],
                    "n_features": n_features,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "avg_nnodes": avg_nnodes,
                    "avg_nleaves": avg_nleaves,
                    "avg_depth": avg_depth,
                    "q": Q,
                    "jaccard": J,
                    "coverage_ratio": coverage_ratio,
                    "mean_distance": mean_dist,
                    "mean_length": mean_length,
                    "init_time": init_time,
                    "runtime": runtime,
                    "clique_size": clique_size,
                    "grown_clique_size": grown_clique_size,
                    "ext_min": ext_min,
                    "ext_avg": ext_avg,
                    "ext_max": ext_max
                }
                results = results.append(run_result, ignore_index=True)
                results.to_csv(run_path + "/" + ds + ".csv", index=False)
                # log progress
                progress_bar.update()
                progress_bar_ds.update()
        progress_bar_ds.close()
    progress_bar.close()
    print("Finished varying ntrees")


def compare_methods(ds_names, explainers=["FACETIndex", "OCEAN"], distance="Euclidean", num_iters=5, eval_samples=20, test_size=0.2, seed=None):
    '''
    Experiment to compare the performanec of different explanation methods
    '''
    # output directory
    run_id, run_path = check_create_directory("./results/compare-methods/")

    # run configuration
    dets = ["RandomForest"]
    agg = "NoAggregator"
    # params = {
    #     "rf_difference": 0.01,
    #     "rf_distance": distance,
    #     "rf_k": 1,
    #     "rf_ntrees": 100,
    #     "rf_threads": 1,
    #     "rf_maxdepth": 5,
    #     "expl_greedy": False,
    #     "expl_distance": distance,
    #     "ocean_norm": 2,
    #     "mace_maxtime": 300,
    #     "num_iters": num_iters,
    #     "eval_samples": eval_samples,
    #     "test_size": test_size,
    #     "facet_graphtype": "disjoint",
    #     "facet_offset": 0.001,
    #     "facet_mode": "exhaustive",
    #     "rf_hardvoting": True,  # TODO consider OCEAN vs FACETIndex soft vs hard requirment,
    #     "facet_sample": "Augment",
    #     "facet_nrects": 20000,
    #     "bb_upperbound": False,
    #     "bb_ordering": "ModifiedPriorityQueue",
    #     "bb_logdists": False,
    #     "verbose": False,
    #     "facet_enumerate": "PointBased",
    #     "bi_nrects": 20000
    # }

    test_params = {
        "num_iters": num_iters,
        "eval_samples": eval_samples,
        "test_size": test_size,
        "explainers": explainers
    }
    rf_params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": 20,
        "rf_threads": 1,
        "rf_maxdepth": 5,
        "rf_hardvoting": True,  # note OCEAN and FACETIndex use soft and hard requirment
    }
    facet_params = {
        "facet_expl_distance": distance,
        "facet_offset": 0.001,
        "facet_verbose": False,
        "facet_sample": "Augment",
        "facet_nrects": 60000,
        "facet_enumerate": "PointBased",
        "bi_nrects": 20000,
        "facet_sd": 0.3,
        "facet_search": "BitVector",
        "rbv_initial_radius": 0.05,
        "rbv_radius_growth": "Linear",
        "rbv_num_interval": 4
    }
    params = {
        "test": test_params,
        "RandomForest": rf_params,
        "FACETIndex": facet_params,
        "ocean_norm": 2
    }

    with open(run_path + "/" + "config.txt", 'a') as f:
        json_text = json.dumps(params, indent=4)
        f.write(json_text)

    # compute the total number of runs for this experiment
    total_runs = len(ds_names) * len(explainers) * num_iters

    # perform the experiment
    print("Comparing Explainers")
    print("\tExplainers:", explainers)
    print("\tDatasets:", ds_names)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=True)

    for ds in ds_names:
        print("DS:", ds)
        # dataframe to store results of each datasets runs
        results = pd.DataFrame(
            columns=[
                "explainer", "n_samples", "n_samples_explained", "n_features", "accuracy", "precision", "recall", "f1", "avg_nnodes", "avg_nleaves", "avg_depth", "q", "jaccard", "coverage_ratio", "mean_distance", "mean_length", "init_time", "runtime", "clique_size", "grown_clique_size", "ext_min", "ext_avg", "ext_max"
            ])
        progress_bar_ds = tqdm(total=len(explainers) * num_iters, desc=ds, leave=False)

        x, y = load_data(ds, normalize=True)
        for i in range(num_iters):
            xtrain, xtest, ytrain, ytest = train_test_split(
                x, y, test_size=test_size, shuffle=True, random_state=seed+i)
            n_samples = DS_DIMENSIONS[ds][0]
            n_features = DS_DIMENSIONS[ds][1]

            if eval_samples is not None:
                xtest = xtest[:eval_samples]
                ytest = ytest[:eval_samples]
                n_samples = eval_samples

            # Create and train the model
            model = HEEAD(detectors=dets, aggregator=agg, hyperparameters=params)
            model.train(xtrain, ytrain)
            preds = model.predict(xtest)

            # compute forest metrics
            accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)
            avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()
            Q, qs = model.detectors[0].compute_qs(xtest, ytest)
            J, jaccards = compute_jaccard(model.detectors[0])

            for expl in explainers:
                # create and prep explainer
                model.set_explainer(expl, hyperparameters=params)

                # FACET Index requires the use of hard-voting which is not supported by OCEAN
                if expl == "FACETIndex" or expl == "FACETBranchBound" or expl == "FACETGrow":
                    model.detectors[0].hard_voting = True
                    # print("hard_voting", expl + " " + str(model.detectors[0].hard_voting))
                elif expl == "OCEAN":
                    model.detectors[0].hard_voting = False
                else:
                    model.detectors[0].hard_voting = False
                preds = model.predict(xtest)

                start_build = time.time()
                model.prepare(data=xtrain)
                end_build = time.time()

                # explain instances
                start = time.time()
                explanations = model.explain(xtest, preds)
                end = time.time()
                runtime = end-start  # wall time in seconds
                init_time = end_build - start_build

                # collect optional statistics
                if expl == "FACETTrees" or expl == "FACETPaths":
                    clique_size = model.explainer.get_clique_size()
                    grown_clique_size = -1
                elif expl == "FACETGrow":
                    clique_size, grown_clique_size = model.explainer.get_clique_size()
                    ext_min = -1
                    ext_avg = -1
                    ext_max = -1
                elif expl == "FACETBranchBound":
                    clique_size = -1
                    grown_clique_size = -1
                    ext_min = model.explainer.ext_min
                    ext_avg = model.explainer.ext_avg
                    ext_max = model.explainer.ext_max
                else:
                    clique_size = -1
                    grown_clique_size = -1
                    ext_min = -1
                    ext_avg = -1
                    ext_max = -1

                # Compute explanation metrics
                coverage_ratio = coverage(explanations)
                mean_dist = average_distance(xtest, explanations, distance_metric="Euclidean")
                mean_length = average_distance(xtest, explanations, distance_metric="FeaturesChanged")

                # Save results
                # save the performance
                run_result = {
                    "explainer": expl,
                    "n_samples": n_samples,
                    "n_samples_explained": xtest.shape[0],
                    "n_features": n_features,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "avg_nnodes": avg_nnodes,
                    "avg_nleaves": avg_nleaves,
                    "avg_depth": avg_depth,
                    "q": Q,
                    "jaccard": J,
                    "coverage_ratio": coverage_ratio,
                    "mean_distance": mean_dist,
                    "mean_length": mean_length,
                    "init_time": init_time,
                    "runtime": runtime,
                    "clique_size": clique_size,
                    "grown_clique_size": grown_clique_size,
                    "ext_min": ext_min,
                    "ext_avg": ext_avg,
                    "ext_max": ext_max
                }
                results = results.append(run_result, ignore_index=True)

                # log progress
                progress_bar.update()
                progress_bar_ds.update()

                # save the results for this ds
                results.to_csv(run_path + "/" + ds + ".csv", index=False)
        progress_bar_ds.close()

    progress_bar.close()
    print("Finished comparing explainers")


def time_cliques(ds_names, ntrees=[1, 5, 10, 15, 20], seed=None):
    # output directory
    run_id, run_path = check_create_directory("./results/time-cliques/")
    distance = "Euclidean"
    num_iters = 1

    # run configuration
    explainers = ["FACETPaths"]
    expl = explainers[0]
    dets = ["RandomForest"]
    agg = "NoAggregator"
    params = {
        "rf_ntrees": 20,
        "rf_threads": 1,
        "rf_maxdepth": 3,
        "expl_distance": distance,
        "num_iters": num_iters,
        "facet_graphtype": "disjoint",
        "facet_offset": 0.001,
        "facet_mode": "exhaustive",
        "rf_difference": 0.01,
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "expl_greedy": False,
    }

    # save the run information
    with open(run_path + "/" + "config.txt", 'a') as f:
        f.write("comparing explanation methods\n\n")
        f.write("iterations: {:d}\n\n".format(num_iters))
        f.write("detectors: ")
        for d in dets:
            f.write(d + ", ")
        f.write("\n")
        f.write("aggregator: " + agg + "\n")
        f.write("explainers: ")
        for e in explainers:
            f.write(e + ", ")
        f.write("\n")
        f.write("hyperparameters{\n")
        for k in params.keys():
            f.write("\t" + k + ": " + str(params[k]) + "\n")
        f.write("}\n")

    # compute the total number of runs for this experiment
    total_runs = len(ds_names) * num_iters * len(ntrees)

    # perform the experiment
    print("Timing Cliques")
    print("\tExplainers:", explainers)
    print("\tDatasets:", ds_names)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0)

    for ds in ds_names:
        print("DS:", ds)

        x, y = load_data(ds, normalize=True)

        file_path = run_path + "/" + ds + ".csv"
        with open(file_path, "a") as results_file:
            results_file.write(
                "rf_ntrees, rf_maxdepth, avg_nnodes, avg_nleaves, avg_depth, init_time, clique_time, nkcliques \n")

        for i in range(num_iters):
            for n in ntrees:
                params["rf_ntrees"] = n
                xtrain, xtest, ytrain, ytest = train_test_split(
                    x, y, test_size=0.2, shuffle=True, random_state=seed)
                n_samples = DS_DIMENSIONS[ds][0]
                n_features = DS_DIMENSIONS[ds][1]

                # Create and train the model
                model = HEEAD(detectors=dets, aggregator=agg, hyperparameters=params)
                model.train(xtrain, ytrain)
                preds = model.predict(xtest)

                # compute forest metrics
                accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)
                avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()
                Q, qs = model.detectors[0].compute_qs(xtest, ytest)
                J, jaccards = compute_jaccard(model.detectors[0])

                model.set_explainer(expl, hyperparameters=params)
                start_build = time.time()
                model.prepare()
                end_build = time.time()
                clique_time, nkcliques = model.explainer.get_clique_stats()
                init_time = end_build - start_build

                # Save results
                with open(file_path, "a") as results_file:
                    results_file.write("{ntrees}, {mdepth}, {nnodes}, {nleaves}, {avdepth}, {itime}, {ctime}, {ncliques} \n".format(
                        ntrees=params["rf_ntrees"],
                        mdepth=params["rf_maxdepth"],
                        nnodes=avg_nnodes,
                        nleaves=avg_nleaves,
                        avdepth=avg_depth,
                        itime=init_time,
                        ctime=clique_time,
                        ncliques=nkcliques
                    ))

                if init_time > 100:
                    break

                # log progress
                progress_bar.update()

        progress_bar.close()

    progress_bar.close()
    print("Finished timing cliques")


def bb_ntrees(ds_names, explainer="FACETBranchBound", distance="Euclidean", num_iters=5, eval_samples=20, test_size=0.2, ntrees=[5, 10, 15, 20], depths=[3], seed=None):
    '''
    Experiment to compare the performanec of different explanation methods
    '''
    # output directory
    run_id, run_path = check_create_directory("./results/bb-ntrees/")

    # run configuration
    max_depth = 3
    dets = ["RandomForest"]
    agg = "NoAggregator"
    params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": ntrees[0],
        "rf_threads": 1,
        "rf_maxdepth": max_depth,
        "expl_greedy": False,
        "expl_distance": distance,
        "num_iters": num_iters,
        "eval_samples": eval_samples,
        "test_size": test_size,
        "facet_offset": 0.001,
        "rf_hardvoting": True,
        "bb_ordering": "ModifiedPriorityQueue",
        "bb_upperbound": False
    }

    # save the run information
    with open(run_path + "/" + "config.txt", 'a') as f:
        f.write("comparing explanation methods\n\n")
        f.write("iterations: {:d}\n\n".format(num_iters))
        f.write("detectors: ")
        for d in dets:
            f.write(d + ", ")
        f.write("\n")
        f.write("aggregator: " + agg + "\n")
        f.write("explainers: " + explainer + "\n")
        f.write("\n")
        f.write("hyperparameters{\n")
        for k in params.keys():
            f.write("\t" + k + ": " + str(params[k]) + "\n")
        f.write("}\n")

    # compute the total number of runs for this experiment
    total_runs = len(ds_names) * num_iters * len(ntrees)

    # perform the experiment
    print("BB Runtime vs Ntrees")
    print("\tDatasets:", ds_names)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for ds in ds_names:
        # dataframe to store results of each datasets runs
        results = pd.DataFrame(
            columns=[
                "explainer",
                "n_trees",
                "max_depth",
                "n_samples",
                "n_samples_explained",
                "n_features",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "q",
                "jaccard",
                "coverage_ratio",
                "mean_distance",
                "mean_length",
                "init_time",
                "runtime",
                "ext_min",
                "ext_avg",
                "ext_max",
                "avg_nnodes",
                "avg_nleaves",
                "avg_depth"
            ])
        progress_bar_ds = tqdm(total=len(ntrees) * num_iters, desc=ds, leave=False, position=1)

        x, y = load_data(ds, normalize=True)
        for i in range(num_iters):
            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=seed)
            n_samples = DS_DIMENSIONS[ds][0]
            n_features = DS_DIMENSIONS[ds][1]

            if eval_samples is not None:
                xtest = xtest[:eval_samples]
                ytest = ytest[:eval_samples]

            for n in ntrees:
                for d in depths:
                    # Create and train the model
                    params["rf_ntrees"] = n
                    params["rf_maxdepth"] = d
                    model = HEEAD(detectors=dets, aggregator=agg, hyperparameters=params, explainer=explainer)
                    model.train(xtrain, ytrain)
                    preds = model.predict(xtest)

                    # compute forest metrics
                    accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)
                    avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()
                    Q, qs = model.detectors[0].compute_qs(xtest, ytest)
                    J, jaccards = compute_jaccard(model.detectors[0])

                    # create and prep explainer
                    start_build = time.time()
                    model.prepare()
                    end_build = time.time()

                    # explain instances
                    start = time.time()
                    explanations = model.explain(xtest, preds)
                    end = time.time()
                    runtime = end-start  # wall time in seconds
                    init_time = end_build - start_build

                    # collect optional statistics
                    clique_size = -1
                    grown_clique_size = -1
                    ext_min = model.explainer.ext_min
                    ext_avg = model.explainer.ext_avg
                    ext_max = model.explainer.ext_max

                    # Compute explanation metrics
                    coverage_ratio = coverage(explanations)
                    mean_dist = average_distance(xtest, explanations, distance_metric="Euclidean")
                    mean_length = average_distance(xtest, explanations, distance_metric="FeaturesChanged")

                    # Save results
                    # save the performance
                    avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()
                    run_result = {
                        "explainer": explainer,
                        "n_trees": n,
                        "max_depth": d,
                        "n_samples": n_samples,
                        "n_samples_explained": xtest.shape[0],
                        "n_features": n_features,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "q": Q,
                        "jaccard": J,
                        "coverage_ratio": coverage_ratio,
                        "mean_distance": mean_dist,
                        "mean_length": mean_length,
                        "init_time": init_time,
                        "runtime": runtime,
                        "ext_min": ext_min,
                        "ext_avg": ext_avg,
                        "ext_max": ext_max,
                        "avg_nnodes": avg_nnodes,
                        "avg_nleaves": avg_nleaves,
                        "avg_depth": avg_depth
                    }
                    results = results.append(run_result, ignore_index=True)

                    # log progress
                    progress_bar.update()
                    progress_bar_ds.update()

                    # save the results for this ds
                    results.to_csv(run_path + "/" + ds + ".csv", index=False)
        progress_bar_ds.close()

    progress_bar.close()
    print("Finished bb ntrees runtime")


def hard_vs_soft(ds_names, num_iters=5, test_size=0.2, ntrees=20, max_depth=3, seed=None):
    run_id, run_path = check_create_directory("./results/hard-soft/")

    results = pd.DataFrame(
        columns=[
            "dataset",
            "n_samples",
            "n_features",
            "n_trees",
            "max_depth",
            "test_size",
            "avg_nnodes",
            "avg_nleaves",
            "avg_depth",
            "accuracy_hard",
            "precision_hard",
            "recall_hard",
            "f1_hard",
            "accuracy_soft",
            "precision_soft",
            "recall_soft",
            "f1_soft",
        ])

    distance = "Euclidean"
    params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": ntrees,
        "rf_maxdepth": max_depth,
        "rf_threads": 8,
        "rf_hardvoting": True,
        "expl_distance": "Euclidean",
        "facet_offset": 0.001
    }

    total_runs = len(ds_names) * num_iters
    progress_bar = tqdm(total=total_runs, desc="Hard vs Soft Progress", position=0, disable=False)
    for ds in ds_names:
        for i in range(num_iters):
            x, y = load_data(ds)
            xtrain, xtest, ytrain, ytest = train_test_split(
                x, y, test_size=test_size, shuffle=True, random_state=seed)

            # Create, train, and predict with the model
            model = HEEAD(detectors=["RandomForest"], aggregator="NoAggregator",
                          explainer="FACETBranchBound", hyperparameters=params)
            model.train(xtrain, ytrain)
            model.prepare()

            # measure model stats
            avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()

            # model performance hard
            preds_hard = model.predict(xtest)
            accuracy_hard, precision_hard, recall_hard, f1_hard = classification_metrics(
                preds_hard, ytest, verbose=False)

            # model performance soft
            model.detectors[0].hard_voting = False
            preds_soft = model.predict(xtest)
            accuracy_soft, precision_soft, recall_soft, f1_soft = classification_metrics(
                preds_soft, ytest, verbose=False)

            run_result = {
                "dataset": ds,
                "n_samples": x.shape[0],
                "n_features": x.shape[1],
                "n_trees": ntrees,
                "max_depth": max_depth,
                "test_size": test_size,
                "avg_nnodes": avg_nnodes,
                "avg_nleaves": avg_nleaves,
                "avg_depth": avg_depth,
                "accuracy_hard": accuracy_hard,
                "precision_hard": precision_hard,
                "recall_hard": recall_hard,
                "f1_hard": f1_hard,
                "accuracy_soft": accuracy_soft,
                "precision_soft": precision_soft,
                "recall_soft": recall_soft,
                "f1_soft": f1_soft,
            }
            results = results.append(run_result, ignore_index=True)
            progress_bar.update()
            results.to_csv(run_path + "/" + ds + ".csv", index=False)
    progress_bar.close()
    print("Finished copmaring hard vs soft voting")


def bb_ordering(ds_names, orderings=["PriorityQueue", "Stack", "Queue"], num_iters=5, test_size=0.2, ntrees=10, max_depth=3, eval_samples=5, seed=None):
    run_id, run_path = check_create_directory("./results/bb-ordering/")

    result_columns = [
        "n_samples",
        "n_samples_explained",
        "n_features",
        "n_trees",
        "max_depth",
        "avg_nnodes",
        "avg_nleaves",
        "avg_depth",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "ext_min",
        "ext_avg",
        "ext_max",
        "lb_min",
        "lb_avg",
        "lb_max",
        "bt_min",
        "bt_avg",
        "bt_max",
        "init_time",
        "runtime",
        "order",
        "coverage_ratio",
        "mean_dist",
        "mean_length",
        "dataset"
    ]

    distance = "Euclidean"
    params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": ntrees,
        "rf_maxdepth": max_depth,
        "rf_threads": 8,
        "rf_hardvoting": True,
        "expl_distance": "Euclidean",
        "facet_offset": 0.001,
        "bb_upperbound": False,
        "bb_ordering": orderings[0],
        "bb_logdists": True
    }

    total_runs = len(ds_names) * num_iters * len(orderings)
    progress_bar = tqdm(total=total_runs, disable=False)
    for ds in ds_names:
        x, y = load_data(ds)
        results = pd.DataFrame(columns=result_columns)
        for i in range(num_iters):
            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size, shuffle=True, random_state=seed)
            # Create, train, and predict with the model
            model = HEEAD(detectors=["RandomForest"], aggregator="NoAggregator",
                          explainer="FACETBranchBound", hyperparameters=params)
            model.train(xtrain, ytrain)
            # create and prep explainer
            start_build = time.time()
            model.prepare(data=xtrain)
            end_build = time.time()

            # measure model performance and stats
            preds = model.predict(xtest)
            avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()
            accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)

            for order in orderings:
                model.explainer.hyperparameters["bb_ordering"] = order
                # explain instances
                if eval_samples is not None:
                    xtest_expl = xtest[:eval_samples]
                    ytest_expl = ytest[:eval_samples]
                    preds_expl = preds[:eval_samples]
                start = time.time()
                explanations = model.explain(xtest_expl, preds_expl)
                end = time.time()
                runtime = end-start  # wall time in seconds
                init_time = end_build - start_build

                if params["bb_logdists"]:
                    # save the distance of solutions found during the optimization process
                    intermediate_dists = model.explainer.intermediate_dists
                    pd.DataFrame(intermediate_dists).to_csv(run_path + "/" + ds + "_dists_" + order.lower() + ".csv")

                # explanation performance and stats
                coverage_ratio = coverage(explanations)
                mean_dist = average_distance(xtest_expl, explanations, distance_metric="Euclidean")
                mean_length = average_distance(xtest_expl, explanations, distance_metric="FeaturesChanged")

                run_result = {
                    "n_samples": x.shape[0],
                    "n_samples_explained": eval_samples,
                    "n_features": x.shape[1],
                    "n_trees": ntrees,
                    "max_depth": max_depth,
                    "avg_nnodes": avg_nnodes,
                    "avg_nleaves": avg_nleaves,
                    "avg_depth": avg_depth,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ext_min": model.explainer.ext_min,
                    "ext_avg": model.explainer.ext_avg,
                    "ext_max": model.explainer.ext_max,
                    "lb_min": model.explainer.lb_min,
                    "lb_avg": model.explainer.lb_avg,
                    "lb_max": model.explainer.lb_max,
                    "bt_min": model.explainer.bt_min,
                    "bt_avg": model.explainer.bt_avg,
                    "bt_max": model.explainer.bt_max,
                    "init_time": init_time,
                    "runtime": runtime,
                    "order": order,
                    "coverage_ratio": coverage_ratio,
                    "mean_dist": mean_dist,
                    "mean_length": mean_length,
                    "dataset": ds
                }
                results = results.append(run_result, ignore_index=True)
                progress_bar.update()
                results.to_csv(run_path + "/" + ds + ".csv", index=False)
    progress_bar.close()
    print("Finished comparing branch and bound orderings")


def index_test(ds_names, exp_var, exp_vals, num_iters=5, eval_samples=20, test_size=0.2, seed=None):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    # output directory
    run_id, run_path = check_create_directory("./results/index-test/")

    # run configuration
    dets = ["RandomForest"]
    agg = "NoAggregator"
    expl = "FACETIndex"
    distance = "Euclidean"

    test_params = {
        "num_iters": num_iters,
        "eval_samples": eval_samples,
        "test_size": test_size,
        "exp_var": exp_var,
        "exp_vals": exp_vals,
    }
    rf_params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": 20,
        "rf_threads": 1,
        "rf_maxdepth": 7,
        "rf_hardvoting": True,  # note OCEAN and FACETIndex use soft and hard requirment
    }
    facet_params = {
        "facet_expl_distance": distance,
        "facet_offset": 0.001,
        "facet_verbose": False,
        "facet_sample": "Augment",
        "facet_nrects": 60000,
        "facet_enumerate": "PointBased",
        "bi_nrects": 20000,
        "facet_sd": 0.3,
        "facet_search": "BitVector",
        "rbv_initial_radius": 0.05,
        "rbv_radius_growth": "Linear",
        "rbv_num_interval": 4
    }
    params = {
        "test": test_params,
        "RandomForest": rf_params,
        "FACETIndex": facet_params
    }

    with open(run_path + "/" + "config.txt", 'a') as f:
        json_text = json.dumps(params, indent=4)
        f.write(json_text)

    # compute the total number of runs for this experiment
    total_runs = len(ds_names) * len(exp_vals) * num_iters
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=True)

    for ds in ds_names:
        progress_bar_ds = tqdm(total=len(exp_vals) * num_iters, desc=ds, leave=False)

        x, y = load_data(ds, normalize=True)
        # dataframe to store results of each datasets runs
        results = pd.DataFrame(columns=["dataset", "n_trees", "explainer", "n_samples", "n_samples_explained", "n_features", "accuracy", "precision", "recall", "f1", "avg_nnodes", "avg_nleaves",
                               "avg_depth", "q", "jaccard", "coverage_ratio", "mean_distance", "mean_length", "init_time", "runtime", "clique_size", "grown_clique_size", "ext_min", "ext_avg", "ext_max", "n_rects", "cover_xtrain", "cover_xtest", "rects_0", "rects_1", "facet_sd", "facet_search", "rbv_initial_radius", "rbv_radius_growth", "rbv_num_interval", "rects_searched_0", "rects_searched_1", "idx_dim_0", "idx_dim_1"])

        for i in range(num_iters):
            # random split the data
            xtrain, xtest, ytrain, ytest = train_test_split(
                x, y, test_size=test_size, shuffle=True, random_state=seed+i)
            n_samples = DS_DIMENSIONS[ds][0]
            n_features = DS_DIMENSIONS[ds][1]

            if eval_samples is not None:
                xtest = xtest[:eval_samples]
                ytest = ytest[:eval_samples]
                n_samples = eval_samples

            # Create and train the model
            model = HEEAD(detectors=dets, aggregator=agg, hyperparameters=params)
            model.set_explainer(expl, hyperparameters=params)
            model.train(xtrain, ytrain)
            preds = model.predict(xtest)

            # compute forest metrics
            accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)
            avg_nnodes, avg_nleaves, avg_depth = model.detectors[0].get_tree_information()
            Q, qs = model.detectors[0].compute_qs(xtest, ytest)
            J, jaccards = compute_jaccard(model.detectors[0])
            for val in exp_vals:
                # update the experimental value
                params[expl][exp_var] = val
                model.set_explainer(expl, hyperparameters=params)

                # explain instances
                start_build = time.time()
                model.prepare(data=xtrain)
                end_build = time.time()
                start = time.time()
                explanations = model.explain(xtest, preds)
                end = time.time()
                runtime = end-start  # wall time in seconds
                init_time = end_build - start_build

                cover_xtrain = model.explainer.explore_index(points=xtrain)
                cover_xtest = model.explainer.explore_index(points=xtest)

                # Compute explanation metrics
                coverage_ratio = coverage(explanations)
                mean_dist = average_distance(xtest, explanations, distance_metric="Euclidean")
                mean_length = average_distance(xtest, explanations, distance_metric="FeaturesChanged")

                if params["FACETIndex"]["facet_search"] == "BitVector":
                    rects_searched_0 = sum(model.explainer.rbvs[0].search_log)
                    rects_searched_1 = sum(model.explainer.rbvs[1].search_log)
                    idx_dim_0 = sum(model.explainer.rbvs[0].indexed_dimensions)
                    idx_dim_1 = sum(model.explainer.rbvs[1].indexed_dimensions)
                else:
                    rects_searched_0, rects_searched_1, idx_dim_0, idx_dim_1 = -1, -1, -1, -1

                # Save results
                run_result = {
                    "dataset": ds,
                    "n_trees": params["RandomForest"]["rf_ntrees"],
                    "explainer": expl,
                    "n_samples": n_samples,
                    "n_samples_explained": xtest.shape[0],
                    "n_features": n_features,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "avg_nnodes": avg_nnodes,
                    "avg_nleaves": avg_nleaves,
                    "avg_depth": avg_depth,
                    "q": Q,
                    "jaccard": J,
                    "coverage_ratio": coverage_ratio,
                    "mean_distance": mean_dist,
                    "mean_length": mean_length,
                    "init_time": init_time,
                    "runtime": runtime,
                    "n_rects": params["FACETIndex"]["facet_nrects"],
                    "cover_xtrain": cover_xtrain,
                    "cover_xtest": cover_xtest,
                    "rects_0": len(model.explainer.index[0]),
                    "rects_1": len(model.explainer.index[1]),
                    "facet_sd": model.explainer.standard_dev,
                    "facet_search": params["FACETIndex"]["facet_search"],
                    "rbv_initial_radius": params["FACETIndex"]["rbv_initial_radius"],
                    "rbv_radius_growth": params["FACETIndex"]["rbv_radius_growth"],
                    "rbv_num_interval": params["FACETIndex"]["rbv_num_interval"],
                    "rects_searched_0": rects_searched_0,
                    "rects_searched_1": rects_searched_1,
                    "idx_dim_0": idx_dim_0,
                    "idx_dim_1": idx_dim_1,
                }
                results = results.append(run_result, ignore_index=True)
                results.to_csv(run_path + "/" + ds + ".csv", index=False)
                # log progress
                progress_bar.update()
                progress_bar_ds.update()
        progress_bar_ds.close()
    progress_bar.close()
    print("Finished varying nrects")
