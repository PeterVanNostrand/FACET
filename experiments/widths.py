import json
import random
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from dataset import load_data
from manager import MethodManager

from .experiments import TUNED_FACET_SD, DEFAULT_PARAMS, FACET_TUNED_M


def compute_widths(ds_names, iteration=0, ntrees=10, max_depth=5):
    '''
    Experiment to compare the performance of different explainers on the same ensemble
    '''
    print("Computing rectangle widths:")
    print("\tds_names:", ds_names)

    experiment_path = "./results/widths/"
    config_path = experiment_path + "widths_config.json"

    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)

    params = DEFAULT_PARAMS
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth

    test_size = 0.2
    preprocessing = "Normalize"

    config = {}
    config["explainer"] = "FACETIndex"
    config["iteration"] = iteration
    config["datasets"] = ds_names
    config["preprocessing"] = preprocessing
    config["test_size"] = test_size
    config["output_path"] = experiment_path
    config["using_tuned_m"] = True
    config["using_tuned_sd"] = True
    config["params"] = params

    with open(config_path, "w") as f:
        json_text = json.dumps(config, indent=4)
        f.write(json_text)

    for ds in ds_names:
        # set appropriate random seeds for reproducibility
        random_state = iteration
        random.seed(random_state)
        np.random.seed(random_state)

        params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds]
        params["FACETIndex"]["rbv_num_interval"] = FACET_TUNED_M[ds]

        x, y = load_data(ds, preprocessing=preprocessing)
        indices = np.arange(start=0, stop=x.shape[0])
        xtrain, xtest, ytrain, ytest, idx_train, idx_test = train_test_split(
            x, y, indices, test_size=test_size, shuffle=True, random_state=random_state)

        # create the manager which handles create the RF model and explainer
        manager: MethodManager = MethodManager(
            explainer="FACETIndex", hyperparameters=params, random_state=random_state)
        manager.train(xtrain, ytrain)
        manager.explainer.prepare_dataset(x, y)
        manager.prepare(xtrain=xtrain, ytrain=ytrain)

        # measure the width of the resulting indices
        index = manager.explainer.index  # list of shape (nclasses, nrects, ndim, 2)
        nrects = len(index[0]) + len(index[1])
        nfeatures = x.shape[1]
        widths = np.empty(shape=(nrects, nfeatures))
        rect_id = 0
        for class_id in [0, 1]:
            for rect in index[class_id]:
                lower_bounds = rect[:, 0]
                upper_bounds = rect[:, 1]
                widths[rect_id] = upper_bounds - lower_bounds
                rect_id += 1

        # save the results to a csv
        csv_path = experiment_path + "widths_{}.csv".format(ds)
        df_widths = pd.DataFrame(widths)
        df_widths.to_csv(csv_path, index=False)

    print("Finished computing widths")
