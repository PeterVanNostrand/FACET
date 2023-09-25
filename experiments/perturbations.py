import json
import random
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from dataset import load_data
from manager import MethodManager
from utilities.metrics import classification_metrics

from .experiments import TUNED_FACET_SD, DEFAULT_PARAMS, FACET_TUNED_M


def perturb_explanations(ds_names, explainers=["FACETIndex", "OCEAN", "RFOCSE", "AFT", "MACE"], nperts=100, pert_sizes=[0.01, 0.05], iterations=[0, 1, 2, 3, 4], fmod=None, ntrees=10, max_depth=5):
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
    else:
        experiment_path = "./results/perturbations/"
    config_path = experiment_path + "perturbations_config.json"

    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)

    params = DEFAULT_PARAMS
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth

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
            # set appropriate random seeds for reproducibility
            random_state = iter
            random.seed(random_state)
            np.random.seed(random_state)

            params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds]
            params["FACETIndex"]["rbv_num_interval"] = FACET_TUNED_M[ds]

            x, y = load_data(ds, preprocessing=preprocessing)
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
            manager: MethodManager = MethodManager(explainer=None, hyperparameters=params, random_state=random_state)

            # train ane evalute the random forest model
            manager.train(xtrain, ytrain)
            preds = manager.predict(xtest)
            accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)

            all_explanations: dict[np.ndarray] = {}

            for expl in explainers:
                # prepare the explainer, handles any neccessary preprocessing
                manager.set_explainer(explainer=expl, random_state=random_state)
                manager.explainer.prepare_dataset(x, y)
                manager.prepare(xtrain=xtrain, ytrain=ytrain)

                # explain the samples using RF predictions (not ground truth)
                explain_preds = manager.predict(x_explain)
                all_explanations[expl] = manager.explain(x_explain, explain_preds, opt_robust=True)

            # generate a set of permuation vectors
            unscaled_perturbations = np.random.rand(nperts, x.shape[1])  # random direction vectors
            normalized_perturbations = normalize(unscaled_perturbations, axis=1)  # normalize them to unit length
            per_valid = {}
            for ps in pert_sizes:
                scaled_perturbations = ps * normalized_perturbations  # scale perturbations to desired size
                for expl in explainers:
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
                    csv_path = experiment_path + "{}_{}_perturbations.csv".format(expl.lower(), ds)
                    if not os.path.exists(csv_path):
                        run_result.to_csv(csv_path, index=False)
                    else:
                        run_result.to_csv(csv_path, index=False, mode="a", header=False,)
                    progress_bar.update()
    progress_bar.close()
    print("Finished perbutation experiment!")
