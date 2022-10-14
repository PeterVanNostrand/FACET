import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from dataset import load_data
from manager import MethodManager
from utilities.metrics import (average_distance, classification_metrics,
                               percent_valid)

from .experiments import (FACET_DEFAULT_PARAMS, FACET_TUNED_M,
                          RF_DEFAULT_PARAMS, TUNED_FACET_SD, execute_run)


def vary_m(ds_names, ms=[2, 4, 6, 8, 10], iterations=[0, 1, 2, 3, 4], fmod=None, ntrees=10, max_depth=5):
    '''
    Experiment to observe the affect of k, the number of explanations requested
    '''
    print("Varying m:")
    print("\tds_names:", ds_names)
    print("\tms:", ms)
    print("\titerations:", iterations)

    if fmod is not None:
        csv_path = "./results/vary_m" + fmod + ".csv"
        experiment_path = "./results/vary-m-" + fmod + "/"
    else:
        csv_path = "./results/vary_m.csv"
        experiment_path = "./results/vary-m/"

    explainer = "FACETIndex"
    params = {
        "RandomForest": RF_DEFAULT_PARAMS,
        "FACETIndex": FACET_DEFAULT_PARAMS,
    }
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth
    params["FACETIndex"]["rbv_num_interval"] = ms[0]

    total_runs = len(ds_names) * len(ms) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for ds in ds_names:
            params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds]
            params["FACETIndex"]["rbv_num_interval"] = FACET_TUNED_M[ds]
            # configure run info
            test_size = 0.2
            n_explain = 20
            random_state = iter
            preprocessing = "Normalize"
            random.seed(random_state)
            np.random.seed(random_state)

            # create the output directory
            output_path = experiment_path
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            # load and split the datset using random state for repeatability. Select samples to explain
            x, y = load_data(ds, preprocessing=preprocessing)
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
            manager.prepare(xtrain=xtrain, ytrain=ytrain)
            prep_end = time.time()
            prep_time = prep_end-prep_start

            explain_preds = manager.predict(x_explain)

            for m in ms:
                # update the bit vector initial radius and rebuild the indices
                params["FACETIndex"]["rbv_num_interval"] = m
                manager.explainer.hyperparameters["FACETIndex"]["rbv_num_interval"] = m

                index_start = time.time()
                manager.explainer.build_bitvectorindex()
                index_end = time.time()
                index_time = index_end - index_start

                run_ext = "m{:03d}_".format(m)
                # store this runs configuration
                config = {}
                config["explainer"] = explainer
                config["iteration"] = iter
                config["dataset_name"] = ds
                config["preprocessing"] = preprocessing
                config["test_size"] = test_size
                config["n_explain"] = n_explain
                config["output_path"] = output_path
                config["random_state"] = random_state
                config["params"] = params
                with open(output_path + "{}_{}_{}{:03d}_config.json".format(ds, explainer.lower(), run_ext, iter), "w") as f:
                    json_text = json.dumps(config, indent=4)
                    f.write(json_text)

                # explain the samples using RF predictions (not ground truth)

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
                    "{}_{}_{}{:03d}_explns.csv".format(ds, explainer.lower(), run_ext, iter)
                expl_df.to_csv(explanation_path, index=False)

                # evalute the quality of the explanations
                per_valid = percent_valid(explanations)
                avg_dist = average_distance(x_explain, explanations, distance_metric="Euclidean")
                avg_length = average_distance(x_explain, explanations, distance_metric="FeaturesChanged")

                # store and return the top level results
                run_result = {
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

                with open(output_path + "{}_{}_{}{:03d}_result.json".format(ds, explainer.lower(), run_ext, iter), "w") as f:
                    json_text = json.dumps(run_result, indent=4)
                    f.write(json_text)

                df_item = {
                    "dataset": ds,
                    "explainer": explainer,
                    "n_trees": params["RandomForest"]["rf_ntrees"],
                    "max_depth": params["RandomForest"]["rf_maxdepth"],
                    "facet_m": m,
                    "iteration": iter,
                    "index_time": index_time,
                    **run_result
                }
                experiment_results = pd.DataFrame([df_item])
                if not os.path.exists(csv_path):
                    experiment_results.to_csv(csv_path, index=False)
                else:
                    experiment_results.to_csv(csv_path, index=False, mode="a", header=False,)

                progress_bar.update()
    progress_bar.close()
    print("Finished varying m")
