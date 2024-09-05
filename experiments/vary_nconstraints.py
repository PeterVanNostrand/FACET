import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np
import random
import json
from baselines.ocean.CounterFactualParameters import FeatureType

from dataset import DataInfo, load_data, rescale_discrete, rescale_numeric
from manager import MethodManager
from utilities.metrics import average_distance, classification_metrics, percent_valid


from .experiments import (FACET_DEFAULT_PARAMS, FACET_TUNED_M,
                          RF_DEFAULT_PARAMS, TUNED_FACET_SD)


constrain_binary = False


def random_constraints(nconstr, ds_info: DataInfo):
    constraints = np.zeros(shape=(ds_info.ncols, 2))
    constraints[:, 0] = -1
    constraints[:, 1] = 2

    one_hot_visited = {}
    col_id = 0

    cols_not_visited = [_ for _ in range(ds_info.ncols)]
    while len(cols_not_visited) > 0 and nconstr > 0:
        # randomly pick a column
        if len(cols_not_visited) != 1:
            col_choice = random.randrange(0, len(cols_not_visited) - 1)
        else:
            col_choice = 0
        col_id = cols_not_visited.pop(col_choice)

        # apply a constraint
        if ds_info.col_types[col_id] == FeatureType.Numeric:
            lower_bound = np.random.uniform(low=0.01, high=0.49, size=1)[0]
            upper_bound = np.random.uniform(low=0.49, high=0.99, size=1)[0]
            constraints[col_id, 0] = lower_bound
            constraints[col_id, 1] = upper_bound
            nconstr -= 1
        elif ds_info.col_types[col_id] == FeatureType.Binary and constrain_binary:
            if col_id in ds_info.reverse_one_hot_schema:
                one_hot_feature = ds_info.reverse_one_hot_schema[col_id]
                if one_hot_feature not in one_hot_visited:
                    # randomly set a column high
                    one_hot_visited[one_hot_feature] = True
                    one_hot_cols = ds_info.one_hot_schema[one_hot_feature]
                    to_constrain = random.choice(one_hot_cols)
                    constraints[to_constrain, 0] = 0.9
                    constraints[to_constrain, 1] = 1.1
                    nconstr -= 1
        elif ds_info.col_types[col_id] == FeatureType.Discrete:
            # randomly pick a discrete val
            vals = ds_info.possible_vals[col_id]
            picked_idx = random.randrange(0, len(vals) - 1)
            picked_val = vals[picked_idx]
            if picked_idx > 2:
                lower_val = vals[picked_idx-2]
            else:
                lower_val = 0.5 * picked_val
            if picked_idx < len(vals) - 3:
                upper_val = vals[picked_idx+2]
            else:
                upper_val = 1.5 * picked_val
            constraints[col_id, 0] = lower_val
            constraints[col_id, 1] = upper_val
            nconstr -= 1
    return constraints


def vary_nconstraints(ds_names, nconstraints=[2, 4, 6, 8, 10], iterations=[0, 1, 2, 3, 4], fmod=None, ntrees=10, max_depth=5):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    print("Varying nconstraints:")
    print("\tds_names:", ds_names)
    print("\tnconstraints:", nconstraints)
    print("\titerations:", iterations)

    if fmod is not None:
        csv_path = "./results/vary_nconstraints_" + fmod + ".csv"
        experiment_path = "./results/vary-nconstraints-" + fmod + "/"
    else:
        csv_path = "./results/vary_nconstraints.csv"
        experiment_path = "./results/vary-nconstraints/"

    explainer = "FACET"
    params = {
        "RandomForest": RF_DEFAULT_PARAMS,
        "FACET": FACET_DEFAULT_PARAMS,
    }
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth

    total_runs = len(ds_names) * len(nconstraints) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for ds in ds_names:
            params["FACET"]["facet_sd"] = TUNED_FACET_SD[ds]
            params["FACET"]["rbv_num_interval"] = FACET_TUNED_M[ds]
            random_state = iter
            output_path = experiment_path
            test_size = 0.2
            n_explain = 20
            dataset_name = ds
            iteration = iter
            run_ext = fmod
            preprocessing = "Normalize"
            model_type = "RandomForest"
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
            config["preprocessing"] = "Normlize"
            config["test_size"] = test_size
            config["n_explain"] = n_explain
            config["output_path"] = output_path
            config["random_state"] = random_state
            config["params"] = params
            with open(output_path + "{}_{}_{}{:03d}_config.json".format(dataset_name, explainer.lower(), run_ext, iteration), "w") as f:
                json_text = json.dumps(config, indent=4)
                f.write(json_text)

            # load and split the datset using random state for repeatability. Select samples to explain

            if preprocessing == "Normalize":
                normalize_numeric = True
                normalize_discrete = True
                do_convert = False
                # MACE requires integer discrete features, this is fine as the RF is the same either way
                # we will later normalize when computing the explanation distance later for comparability
                if explainer == "MACE":
                    normalize_discrete = False
                    if dataset_name == "adult":  # will treat numeric-real as numeric-int
                        normalize_numeric = False  # this handles a pysmt bug with mixed-numeric and non-numeric
                        do_convert = True
                if explainer == "RFOCSE":
                    normalize_discrete = False
                    normalize_numeric = True

            x, y, ds_info = load_data(dataset_name, normalize_numeric, normalize_discrete, do_convert)
            indices = np.arange(start=0, stop=x.shape[0])
            xtrain, xtest, ytrain, ytest, idx_train, idx_test = train_test_split(
                x, y, indices, test_size=test_size, shuffle=True, random_state=random_state)

            if n_explain is not None:
                x_explain = xtest[:n_explain]
                # y_explain = ytest[:n_explain]
                idx_explain = idx_test[:n_explain]
            else:
                x_explain = xtest
                # y_explain = ytest
                idx_explain = idx_test
                n_explain = x_explain.shape[0]

            # create the manager which handles create the RF model and explainer
            manager = MethodManager(explainer=explainer, hyperparameters=params,
                                    model_type=model_type)

            # train ane evalute the random forest model
            manager.train(xtrain, ytrain)
            preds = manager.predict(xtest)
            accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)

            # prepare the explainer, handles any neccessary preprocessing
            prep_start = time.time()
            manager.explainer.prepare_dataset(x, y, ds_info)
            manager.prepare(xtrain=xtrain, ytrain=ytrain)
            prep_end = time.time()
            prep_time = prep_end-prep_start

            for nconstr in nconstraints:
                # randomly select a constraints set
                # split the number of constraints evenly between lower and upper
                constraints = random_constraints(nconstr, ds_info)

                # explain the samples using RF predictions (not ground truth)
                explain_preds = manager.predict(x_explain)
                explain_start = time.time()
                explanations: np.ndarray = manager.explain(x_explain, explain_preds, constraints=constraints)

                explain_end = time.time()
                explain_time = explain_end - explain_start
                sample_time = explain_time / n_explain

                # check that the returned explanations fit the data type requirements (one-hot, discrete, binary, etc)
                # if not ds_info.check_valid(explanations):
                #     print("WARNING - {} PRODUCED AN EXPLANATION INCOMPATIBLE WITH THE GIVEN DATA SCHEMA".format(explainer))

                # store the returned explantions
                expl_df = pd.DataFrame(ds_info.unscale(explanations), columns=ds_info.col_names)
                # also store the index of the explained sample in the dataset
                expl_df.insert(0, "x_idx", idx_explain)
                explanation_path = output_path + \
                    "{}_{}_{}{:03d}_explns.csv".format(dataset_name, explainer.lower(), run_ext, iteration)
                expl_df.to_csv(explanation_path, index=False)

                x_df = pd.DataFrame(ds_info.unscale(x_explain), columns=ds_info.col_names)
                x_df.insert(0, "x_idx", idx_explain)
                x_path = output_path + \
                    "{}_{}_{}{:03d}_x.csv".format(dataset_name, explainer.lower(), run_ext, iteration)
                x_df.to_csv(x_path, index=False)

                per_valid = percent_valid(explanations)

                # handle special mace int encoding
                if ds_info.numeric_int_map is not None:
                    print("unscaled")
                    for i in range(x_explain.shape[0]):
                        for col_name in ds_info.numeric_int_map.keys():
                            col_id = ds_info.col_names.index(col_name)
                            expl_val = np.floor(explanations[i][col_id])
                            if expl_val not in [np.inf, -np.inf]:
                                expl_val = int(expl_val)
                                explanations[i][col_id] = ds_info.numeric_int_map[col_name][expl_val]
                            x_val = int(np.floor(x_explain[i][col_id]))
                            explanations[i][col_id] = ds_info.numeric_int_map[col_name][x_val]

                # if we didn't normalize the data we can't trust the distances
                if not ds_info.normalize_numeric or not ds_info.normalize_discrete:
                    # create copies so we don't disturb the underlying data
                    x_explain = x_explain.copy()
                    explanations = explanations.copy()
                    # if we didn't normalize the numeric features, scale them down now
                    if not ds_info.normalize_numeric:
                        ds_info.normalize_numeric = True
                        x_explain = rescale_numeric(x_explain, ds_info, scale_up=False)
                        explanations = rescale_numeric(explanations, ds_info, scale_up=False)
                        ds_info.normalize_numeric = False
                    if not ds_info.normalize_discrete:
                        ds_info.normalize_discrete = True
                        x_explain = rescale_discrete(x_explain, ds_info, scale_up=False)
                        explanations = rescale_discrete(explanations, ds_info, scale_up=False)
                        ds_info.normalize_discrete = False

                # evalute the quality of the explanations
                avg_dist = average_distance(x_explain, explanations, distance_metric="Euclidean")  # L2 Norm Euclidean
                avg_manhattan = average_distance(
                    x_explain, explanations, distance_metric="Manhattan")  # L1 Norm Manhattan
                avg_length = average_distance(x_explain, explanations,
                                              distance_metric="FeaturesChanged")  # L0 Norm Sparsity

                # store and return the top level results
                run_result = {
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
                    "nconstr": nconstr,
                }

                with open(output_path + "{}_{}_{}{:03d}_result.json".format(dataset_name, explainer.lower(), run_ext, iteration), "w") as f:
                    json_text = json.dumps(run_result, indent=4)
                    f.write(json_text)
                df_item = {
                    "dataset": ds,
                    "explainer": explainer,
                    "n_trees": params["RandomForest"]["rf_ntrees"],
                    "max_depth": params["RandomForest"]["rf_maxdepth"],
                    "iteration": iter,
                    **run_result
                }
                experiment_results = pd.DataFrame([df_item])
                if not os.path.exists(csv_path):
                    experiment_results.to_csv(csv_path, index=False)
                else:
                    experiment_results.to_csv(csv_path, index=False, mode="a", header=False,)
                progress_bar.update()
    progress_bar.close()
    print("Finished varying number constraints")
