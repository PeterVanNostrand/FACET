import json
import random

import numpy as np
from sklearn.model_selection import train_test_split

from dataset import DataInfo, load_data
from experiments.experiments import DEFAULT_PARAMS, FACET_TUNED_M, TUNED_FACET_SD
from explainers.facet import FACET
from manager import MethodManager


def flask_setup_server(
    dataset_name,
    explainer,
    params,
    random_state=None,
    preprocessing="Normalize",
):
    random.seed(random_state)
    np.random.seed(random_state)

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

    # Load and split the dataset into train/explain
    x, y, ds_info = load_data(dataset_name, normalize_numeric, normalize_discrete, do_convert)
    indices = np.arange(start=0, stop=x.shape[0])
    xtrain, xtest, ytrain, ytest, idx_train, idx_test = train_test_split(
        x, y, indices, test_size=0.1, shuffle=True, random_state=random_state
    )

    # Create the manager which handles creating the RF model and explainer
    manager = MethodManager(
        explainer=explainer, hyperparameters=params, random_state=random_state
    )
    manager.train(xtrain, ytrain)
    manager.explainer.prepare_dataset(x, y, ds_info)
    manager.prepare(xtrain=xtrain, ytrain=xtrain)

    # get the negative outcome samples for explanation
    preds = manager.predict(xtrain)
    neg_samples = xtrain[preds == 0]

    return manager, neg_samples


def run_facet(
    ds_name="loans",
    explainer="FACET",
    random_state=0,
    ntrees=10,
    max_depth=5,
) -> tuple[MethodManager, FACET]:
    params = DEFAULT_PARAMS
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth
    params["RandomForest"]["rf_maxdepth"] = max_depth
    params["FACET"]["facet_sd"] = TUNED_FACET_SD[ds_name]
    params["FACET"]["rbv_num_interval"] = FACET_TUNED_M[ds_name]

    print("dataset: " + ds_name)

    manager, sample_data = flask_setup_server(
        dataset_name=ds_name,
        explainer=explainer,
        params=params,
        random_state=random_state,
    )

    return manager, sample_data


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

    # create a data info object, app treats all feature as numeric
    ds_info = DataInfo.generic(ncols=ncols)
    ds_info.col_names = col_names
    ds_info.normalize_numeric = is_normalized
    ds_info.col_scales = col_scales
    ds_info.__post_init__()
    ds_info._map_cols()

    # ds_info = DataInfo(ncols=ncols, is_normalized=is_normalized, col_scales=col_scales, col_names=col_names)
    return ds_info
