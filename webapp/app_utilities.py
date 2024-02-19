import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
from manager import MethodManager
from dataset import load_data, DataInfo
from experiments.experiments import DEFAULT_PARAMS, TUNED_FACET_SD, FACET_TUNED_M


def flask_setup_server(
    dataset_name,
    explainer,
    params,
    random_state=None,
    preprocessing="Normalize",
):
    random.seed(random_state)
    np.random.seed(random_state)

    # Load and split the dataset into train/explain
    x, y, min_value, max_value = load_data(dataset_name, preprocessing=preprocessing)
    indices = np.arange(start=0, stop=x.shape[0])
    xtrain, xtest, ytrain, ytest, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=0.1, shuffle=True, random_state=random_state
    )

    # Create the manager which handles creating the RF model and explainer
    manager = MethodManager(
        explainer=explainer, hyperparameters=params, random_state=random_state
    )
    manager.train(xtrain, ytrain)
    manager.explainer.prepare_dataset(x, y)
    manager.prepare(xtrain=xtrain, ytrain=xtrain)

    return manager, xtest


def run_facet(
    ds_name="loans",
    explainer="FACETIndex",
    random_state=0,
    ntrees=10,
    max_depth=5,
):
    params = DEFAULT_PARAMS
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth
    params["RandomForest"]["rf_maxdepth"] = max_depth
    params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds_name]
    params["FACETIndex"]["rbv_num_interval"] = FACET_TUNED_M[ds_name]

    print("dataset: " + ds_name)

    manager, test_applicants = flask_setup_server(
        dataset_name=ds_name,
        explainer=explainer,
        params=params,
        random_state=random_state,
    )

    return manager, test_applicants


def parse_dataset_info(details_path):
    with open(details_path, "r") as details_file:
        ds_details = json.load(details_file)
    is_normalized = ds_details["normalized"]
    ncols = ds_details["n_features"]

    col_to_idx = dict()
    for i in range(ncols):
        col_to_idx["x{}".format(i)] = i

    # create a dict of pairs [min, max] for each feature
    col_scales = dict()
    for col_id in range(ncols):
        col_scales[col_id] = [None, None]
    # get the min vals for each feature
    for col_id, min_val in ds_details["min_values"].items():
        col_scales[col_to_idx[col_id]][0] = min_val
    # get the max vals for each feature
    for col_id, max_val in ds_details["max_values"].items():
        col_scales[col_to_idx[col_id]][1] = max_val
    # get the column names
    col_names = dict()
    for col_id, name in ds_details["feature_names"].items():
        col_names[col_id] = name

    ds_info = DataInfo(ncols=ncols, is_normalized=is_normalized, col_scales=col_scales, col_names=col_names)
    return ds_info
