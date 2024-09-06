import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from baselines.ocean.CounterFactualParameters import FeatureActionability, FeatureType
from dataset import load_data
from manager import MethodManager
from utilities.metrics import classification_metrics

from .experiments import DEFAULT_PARAMS, FACET_TUNED_M, FACET_TUNED_NRECTS, TUNED_FACET_SD


def perturb_explanations(ds_names, explainers=["FACET", "OCEAN", "RFOCSE", "AFT", "MACE"], nperts=100, pert_sizes=[0.01, 0.05], iterations=[0, 1, 2, 3, 4], fmod=None, ntrees=10, max_depth=5, max_time=None):
    '''
    Experiment to compare the performance of different explainers on the same ensemble
    '''
    print("Simulating perbutation:")
    print("\tds_names:", ds_names)
    print("\texplainers:", explainers)
    print("\titerations:", iterations)
    print("\tperturbations:", pert_sizes)

    if fmod is not None:
        experiment_path = "./results/perturbations-" + fmod + "/"
        csv_path = "./results/perturbations_{}.csv".format(fmod)
    else:
        experiment_path = "./results/perturbations/"
        csv_path = "./results/perturbations.csv"
    config_path = experiment_path + "perturbations_config.json"

    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)

    params = DEFAULT_PARAMS
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth
    params["OCEAN"]["ocean_ilf"] = False

    # set max time for slow methods
    params["RFOCSE"]["rfoce_maxtime"] = max_time
    params["MACE"]["mace_maxtime"] = max_time

    n_explain = 1
    test_size = 0.2
    preprocessing = "Normalize"

    config = {}
    config["explainers"] = explainers
    config["iterations"] = iterations
    config["datasets"] = ds_names
    config["preprocessing"] = preprocessing
    config["test_size"] = test_size
    config["n_explain"] = n_explain
    config["output_path"] = experiment_path
    config["using_tuned_m"] = True
    config["using_tuned_sd"] = True
    config["params"] = params

    with open(config_path, "w") as f:
        json_text = json.dumps(config, indent=4)
        f.write(json_text)

    total_runs = len(iterations) * len(ds_names) * len(explainers) * len(pert_sizes)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)
    results = pd.DataFrame(columns=["iteration", "explainer", "datset", "instance",
                           "perturbation_size", "nperturbations", "per_valid"])

    # for each iteration
    for iter in iterations:
        for ds in ds_names:
            for expl in explainers:
                # set appropriate random seeds for reproducibility
                random_state = iter
                random.seed(random_state)
                np.random.seed(random_state)

                params["FACET"]["facet_sd"] = TUNED_FACET_SD[ds]
                params["FACET"]["rbv_num_interval"] = FACET_TUNED_M[ds]
                params["FACET"]["facet_nrects"] = FACET_TUNED_NRECTS[ds]

                if preprocessing == "Normalize":
                    normalize_numeric = True
                    normalize_discrete = True
                    do_convert = False
                    # MACE requires integer discrete features, this is fine as the RF is the same either way
                    # we will later normalize when computing the explanation distance later for comparability
                    if expl == "MACE":
                        normalize_discrete = False
                        if ds == "adult":  # will treat numeric-real as numeric-int
                            normalize_numeric = False  # this handles a pysmt bug with mixed-numeric and non-numeric
                            do_convert = True
                    if expl == "RFOCSE":
                        normalize_discrete = False
                        normalize_numeric = True

                x, y, ds_info = load_data(ds, normalize_numeric, normalize_discrete, do_convert)
                # for col in range(ds_info.ncols):
                #     ds_info.col_types[col] = FeatureType.Numeric
                #     ds_info.col_actions[col] = FeatureActionability.Free
                #     ds_
                indices = np.arange(start=0, stop=x.shape[0])
                xtrain, xtest, ytrain, ytest, idx_train, idx_test = train_test_split(
                    x, y, indices, test_size=test_size, shuffle=True, random_state=random_state)

                if n_explain is not None:
                    x_explain = xtest[:n_explain]
                    y_explain = ytest[:n_explain]
                    idx_explain = idx_test[:n_explain]
                else:
                    x_explain = xtest
                    y_explain = ytest
                    idx_explain = idx_test
                    n_explain = x_explain.shape[0]

                # create the manager which handles create the RF model and explainer
                manager: MethodManager = MethodManager(
                    explainer=None, hyperparameters=params, random_state=random_state)

                # train ane evalute the random forest model
                manager.train(xtrain, ytrain)
                preds = manager.predict(xtest)
                accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)

                all_explanations: dict[np.ndarray] = {}

                # prepare the explainer, handles any neccessary preprocessing
                manager.set_explainer(explainer=expl, random_state=random_state)
                manager.explainer.prepare_dataset(x, y, ds_info)
                manager.prepare(xtrain=xtrain, ytrain=ytrain)

                # explain the samples using RF predictions (not ground truth)
                explain_preds = manager.predict(x_explain)
                all_explanations[expl] = manager.explain(x_explain, explain_preds, opt_robust=True)

                # generate a set of permuation vectors
                unscaled_perturbations = np.random.rand(nperts, x.shape[1])  # random direction vectors
                normalized_perturbations = normalize(unscaled_perturbations, axis=1)  # normalize them to unit length
                per_valid = {}
                for ps in pert_sizes:
                    ds_info
                    scaled_perturbations = ps * normalized_perturbations  # scale perturbations to desired size

                    # for each column
                    for i in range(ds_info.ncols):
                        col_type = ds_info.col_types[i]
                        min_value, max_value = ds_info.col_scales[i]
                        # if the column is not normalized
                        if (col_type == FeatureType.Discrete and not ds_info.normalize_discrete) or \
                                (col_type == FeatureType.Numeric and not ds_info.normalize_numeric):
                            # adjust the perturbation along that column to be proportionally bigger
                            scaled_perturbations[:, i] = scaled_perturbations[:, i] * \
                                (max_value - min_value) + min_value

                    # for expl in explainers:
                    # perturb the sample and record
                    perturbed_explanations = all_explanations[expl] + scaled_perturbations

                    # temporarily swap out failed explanations for prediction
                    if (perturbed_explanations == np.inf).any():
                        per_valid = -1
                    else:
                        # compute the percent valid perturbed explanations
                        perturbed_preds = manager.predict(perturbed_explanations)
                        per_valid = (perturbed_preds != y_explain).sum() / nperts

                    # store results so far
                    row = {
                        "iteration": iter,
                        "explainer": expl,
                        "dataset": ds,
                        "instance": idx_explain[0],
                        "perturbation_size": ps,
                        "nperturbations": nperts,
                        "per_valid": per_valid,
                    }
                    results = results.append(row, ignore_index=True)
                    run_result = pd.DataFrame(row, index=[0])
                    if not os.path.exists(csv_path):
                        run_result.to_csv(csv_path, index=False)
                    else:
                        run_result.to_csv(csv_path, index=False, mode="a", header=False,)
                    progress_bar.update()
    progress_bar.close()
    print("Finished perbutation experiment!")
