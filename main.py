import numpy as np
from numpy.core.fromnumeric import var
from heead import HEEAD
import matplotlib.pyplot as plt
from utilities.metrics import coverage
from utilities.metrics import classification_metrics
from utilities.metrics import average_distance
from utilities.tree_tools import compute_jaccard
from dataset import load_data
from dataset import DS_NAMES
from experiments import *
import cProfile
import time
import random
import math
import json


def simple_run(dataset_name):
    # Load the dataset
    x, y = load_data(dataset_name)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

    # Euclidean, FeaturesChanged
    distance = "Euclidean"
    rf_params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": 10,
        "rf_threads": 1,
        "rf_maxdepth": 20,
        "rf_hardvoting": True,  # note OCEAN and FACETIndex use soft and hard requirment
    }
    facet_params = {
        "facet_expl_distance": distance,
        "facet_offset": 0.001,
        "facet_verbose": False,
        "facet_sample": "Augment",
        "facet_nrects": 10000,
        "facet_enumerate": "PointBased",
        "bi_nrects": 20000,
        "facet_sd": 0.3,
        "facet_search": "BitVector",
        "rbv_initial_radius": 0.01,
        "rbv_radius_growth": "Linear",
        "rbv_num_interval": 4
    }
    params = {
        "RandomForest": rf_params,
        "FACETIndex": facet_params,
        "mace_maxtime": 300,
        "rfoce_transform": False
    }
    json_text = json.dumps(params, indent=4)
    print(json_text)

    # Create, train, and predict with the model
    expl = "RFOCSE"
    model = HEEAD(detectors=["RandomForest"], aggregator="NoAggregator",
                  explainer=expl, hyperparameters=params)
    model.train(xtrain, ytrain)
    prep_start = time.time()
    #! TEMP SWAP TO ALL DATA FOR RFOCSE
    # model.prepare(data=xtrain)
    model.prepare(data=x)
    # if params["rfoce_transform"]:
    #     model.train(model.explainer.float_transformer.transform(xtrain), ytrain)
    # else:
    #     model.train(xtrain, ytrain)

    if expl == "FACETIndex":
        print("rects requested:", params.get("FACETIndex").get("facet_nrects"))
        print("rects enumerated")
        print("\tclass 0:", len(model.explainer.index[0]))
        print("\tclass 1:", len(model.explainer.index[1]))
    # cover_xtrain = model.explainer.explore_index(points=xtrain)
    # cover_xtest = model.explainer.explore_index(points=xtest)
    # print("xtrain index coverage:", cover_xtrain)
    # print("xtest index coverage:", cover_xtest)
    prep_end = time.time()
    preptime = prep_end-prep_start
    print("preptime:", preptime)
    print("xtrain:", xtrain.shape)

    if params["rfoce_transform"]:
        preds = model.predict(model.explainer.float_transformer.transform(xtest))
    else:
        preds = model.predict(xtest)

    # measure model performance
    accuracy, precision, recall, f1 = classification_metrics(preds, ytest, verbose=False)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)

    # Q-stastic
    # Q, qs = model.detectors[0].compute_qs(xtest, ytest)
    # print("Q-Statistic:", Q)

    # jaccard similarity
    # J, jaccards = compute_jaccard(model.detectors[0])
    # print("Jaccard Index:", J)

    # generate the explanations
    explain = True
    eval_samples = None
    if explain:
        if eval_samples is not None:
            xtest = xtest[:eval_samples]
            preds = preds[:eval_samples]

        start = time.time()
        explanations = model.explain(xtest, preds)
        end = time.time()
        runtime = end-start
        print("runtime:", runtime)

        if eval_samples is not None:
            sample_time = runtime / eval_samples
            print("sample_time:", sample_time)

        coverage_ratio = coverage(explanations)
        print("coverage_ratio:", coverage_ratio)
        mean_dist = average_distance(xtest, explanations, distance_metric="Euclidean")
        print("mean_dist:", mean_dist)
        mean_length = average_distance(xtest, explanations, distance_metric="FeaturesChanged")
        print("mean_length", mean_length)

        if False:
            ext_min = model.explainer.ext_min
            ext_avg = model.explainer.ext_avg
            ext_max = model.explainer.ext_max

            print("ext_min:", ext_min)
            print("ext_avg:", ext_avg)
            print("ext_max:", ext_max)


if __name__ == "__main__":
    RAND_SEED = 0
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    run_ds = DS_NAMES.copy()
    # index_test(ds_names=["vertebral"], exp_var="facet_nrects", exp_vals=[1000, 5000, 10000],
    #            num_iters=1, eval_samples=20, test_size=0.2, seed=RAND_SEED)

    # index_test(ds_names=run_ds, exp_var="facet_intersect_order", exp_vals=["Axes", "Size", "Ensemble"],
    #            num_iters=1, eval_samples=20, test_size=0.2, seed=RAND_SEED)
    # index_test()
    # run_ds.remove("spambase")
    # compare_methods(["vertebral"], num_iters=1, explainers=["MACE"], eval_samples=20, seed=RAND_SEED)
    # vary_ntrees(run_ds, explainer="FACETIndex", ntrees=list(range(5, 105, 5)), num_iters=5, seed=SEED)
    simple_run("vertebral")
    # bb_ntrees(run_ds, ntrees=[25], depths=[3], num_iters=1, eval_samples=5)
    # hard_vs_soft(run_ds, num_iters=10)
    # bb_ordering(run_ds, orderings=["PriorityQueue", "Stack", "ModifiedPriorityQueue"], num_iters=1,
    # test_size=0.2, ntrees=15, max_depth=3, eval_samples=5)
    # 100, 1000, 5000, 10000, 20000, 30000, 40000, 50000
    # vary_nrects(run_ds, nrects=[20000, 40000, 60000, 80000, 100000, 150000,
    # 200000, 250000], num_iters=1, eval_samples=20, seed=RAND_SEED)
