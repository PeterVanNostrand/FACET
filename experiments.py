import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from heead import HEEAD
from dataset import load_data

from utilities.metrics import coverage
from utilities.metrics import classification_metrics
from utilities.metrics import mean_distance


def check_create_directory(dir_path="./results"):
    '''
    Checks the the directory at `dir_path` exists, if it does not it creates all directories in the path
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


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

    # save the performance
    run_perf = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage_ratio": coverage_ratio,
        "mean_distance": mean_dist
    }

    return run_perf


def vary_difference():
    '''
    Experiment to observe the effect of the difference value using in decision tree explanation
    '''
    for ds_name in ["cardio", "musk", "thyroid", "wbc"]:
        # Load the dataset
        x, y = load_data(ds_name, normalize=True)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

        # dataframe to store results of all experimental runs
        results = pd.DataFrame(columns=["difference", "accuracy", "precision",
                                        "recall", "f1", "coverage_ratio", "mean_distance"])
        check_create_directory("./results/vary-difference/")

        differences = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for diff in differences:
            # Create, train, and predict with the model
            params = {
                "rf_difference": diff,
                "rf_distance": "Euclidean",
                "rf_k": 1,
                "rf_ntrees": 20
            }
            model = HEEAD(detectors=["RandomForest"], aggregator="LogisticRegression",
                          explainer="BestCandidate", hyperparameters=params)
            diff_val = {"difference": diff}
            run_perf = execute_run(model, xtrain, xtest, ytrain, ytest)
            run_result = {**diff_val, **run_perf}
            results = results.append(run_result, ignore_index=True)

        # save the results
        results.to_csv("./results/vary-difference/" + ds_name + ".csv")


def vary_k():
    '''
    Experiment to observe the effect of the difference value using in decision tree explanation
    '''
    for ds_name in ["cardio", "musk", "thyroid", "wbc"]:
        # Load the dataset
        x, y = load_data(ds_name, normalize=True)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

        # dataframe to store results of all experimental runs
        results = pd.DataFrame(columns=["k", "accuracy", "precision",
                                        "recall", "f1", "coverage_ratio", "mean_distance"])
        check_create_directory("./results/vary-k/")

        ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]

        for k in ks:
            # Create, train, and predict with the model
            params = {
                "rf_difference": 0.01,
                "rf_distance": "Euclidean",
                "rf_k": k,
                "explainer_k": k,
                "rf_ntrees": 20
            }
            model = HEEAD(detectors=["RandomForest"], aggregator="LogisticRegression",
                          explainer="BestCandidate", hyperparameters=params)
            diff_val = {"k": k}
            run_perf = execute_run(model, xtrain, xtest, ytrain, ytest)
            run_result = {**diff_val, **run_perf}
            results = results.append(run_result, ignore_index=True)

        # save the results
        results.to_csv("./results/vary-k/" + ds_name + ".csv")


def vary_dim():
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    for ds_name in ["cardio", "musk", "thyroid", "wbc"]:
        num_iters = 5
        runs_complete = 0

        # dataframe to store results of each datasets runs
        results = pd.DataFrame(columns=["n_features", "accuracy", "precision",
                               "recall", "f1", "coverage_ratio", "mean_distance"])
        check_create_directory("./results/vary-dim/")

        for i in range(num_iters):
            # Load the dataset
            x, y = load_data(ds_name, normalize=True)

            # randomly shuffle the order of the columns
            np.random.shuffle(np.transpose(x))

            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

            # try first 1 feature, first 2 feature, ... , first n features
            for n in range(2, x.shape[1]):
                # Create, train, and predict with the model
                params = {
                    "rf_difference": 0.01,
                    "rf_distance": "Euclidean",
                    "rf_k": 1,
                    "rf_ntrees": 20
                }
                model = HEEAD(detectors=["RandomForest"], aggregator="LogisticRegression",
                              explainer="BestCandidate", hyperparameters=params)
                diff_val = {"n_features": n}
                run_perf = execute_run(model, xtrain[:, :n], xtest[:, :n], ytrain, ytest)
                run_result = {**diff_val, **run_perf}
                results = results.append(run_result, ignore_index=True)
                runs_complete += 1
                print("runs complete:", runs_complete)

        print("finished", ds_name)

        # save the results
        results.to_csv("./results/vary-dim/" + ds_name + ".csv")


def vary_ntrees():
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    for ds_name in ["cardio", "musk", "thyroid", "wbc"]:
        num_iters = 5
        runs_complete = 0

        # dataframe to store results of each datasets runs
        results = pd.DataFrame(columns=["n_trees", "accuracy", "precision",
                               "recall", "f1", "coverage_ratio", "mean_distance"])
        check_create_directory("./results/vary-ntrees/")

        for i in range(num_iters):
            # Load the dataset
            x, y = load_data(ds_name, normalize=True)
            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True)

            max_trees = 100
            for n in range(1, max_trees):
                # Create, train, and predict with the model
                params = {
                    "rf_difference": 0.01,
                    "rf_distance": "Euclidean",
                    "rf_k": 1,
                    "rf_ntrees": n
                }
                model = HEEAD(detectors=["RandomForest"], aggregator="LogisticRegression",
                              explainer="BestCandidate", hyperparameters=params)
                diff_val = {"n_trees": n}
                run_perf = execute_run(model, xtrain[:, :n], xtest[:, :n], ytrain, ytest)
                run_result = {**diff_val, **run_perf}
                results = results.append(run_result, ignore_index=True)
                runs_complete += 1
                print("runs complete:", runs_complete)

        print("finished", ds_name)

        # save the results
        results.to_csv("./results/vary-ntrees/" + ds_name + ".csv")
