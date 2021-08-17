import os
import re

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

from heead import HEEAD
from dataset import load_data
from dataset import DS_DIMENSIONS

from utilities.metrics import coverage
from utilities.metrics import classification_metrics
from utilities.metrics import mean_distance


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


def execute_run(model, xtrain, xtest, ytrain, ytest):
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
    model.train(xtrain, ytrain)
    preds = model.predict(xtest)

    # generate the explanations
    explanations = model.explain(xtest, preds)

    # measure model performance
    accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)
    coverage_ratio = coverage(explanations)
    mean_dist = mean_distance(xtest, explanations)
    mean_length = (xtest == explanations).sum(axis=1).mean()

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
    expl = "BestCandidate"
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
    expl = "BestCandidate"
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


def vary_dim(ds_names, explainer):
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
        "rf_distance": "Euclidean",
        "rf_k": 1,
        "rf_ntrees": 20
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

        for i in range(num_iters):
            # Load the dataset
            x, y = load_data(ds, normalize=True)

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
                J, jaccards = model.detectors[0].compute_jaccard()
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


def vary_ntrees():
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    # output directory
    run_id, run_path = check_create_directory("./results/vary-ntrees/")

    # run configuration
    min_trees = 1
    max_trees = 100
    num_iters = 10
    dets = ["RandomForest"]
    agg = "LogisticRegression"
    expl = "BestCandidate"
    params = {
        "rf_difference": 0.01,
        "rf_distance": "Euclidean",
        "rf_k": 1,
        "rf_ntrees": min_trees
    }

    # save the run information
    with open(run_path + "/" + "config.txt", 'a') as f:
        f.write("varying the number of trees\n\n")
        f.write("min trees: {:d}\n".format(min_trees))
        f.write("max trees: {:d}\n".format(max_trees))
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

    for ds_name in ["thyroid", "cardio", "wbc", "musk"]:
        x, y = load_data(ds_name, normalize=True)
        # dataframe to store results of each datasets runs
        results = pd.DataFrame(columns=["n_trees", "accuracy", "precision",
                               "recall", "f1", "coverage_ratio", "mean_distance"])
        runs_complete = 0

        for n in range(min_trees, max_trees + 1):
            params["rf_ntrees"] = n
            for i in range(num_iters):
                # random split the data
                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

                # Create, train, and predict with the model
                model = HEEAD(detectors=dets, aggregator=agg, explainer=expl, hyperparameters=params)
                run_perf = execute_run(model, xtrain[:, :n], xtest[:, :n], ytrain, ytest)

                # store results
                diff_val = {"n_trees": n}
                run_result = {**diff_val, **run_perf}
                results = results.append(run_result, ignore_index=True)

                # log progress
                runs_complete += 1
                print("\truns complete:", runs_complete)
        # save the results for this dataset
        results.to_csv(run_path + "/" + ds_name + ".csv")
        print("finished", ds_name)
