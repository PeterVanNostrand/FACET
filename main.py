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


def simple_run(dataset_name):
    # Load the dataset
    x, y = load_data(dataset_name)

    # Euclidean, FeaturesChanged
    distance = "Euclidean"
    params = {
        "rf_difference": 0.01,
        "rf_distance": distance,
        "rf_k": 1,
        "rf_ntrees": 20,
        "rf_maxdepth": 5,
        "rf_threads": 8,
        "expl_greedy": False,
        "expl_distance": distance,
        "facet_graphtype": "Disjoint"
    }

    # Create, train, and predict with the model
    model = HEEAD(detectors=["RandomForest"], aggregator="NoAggregator", explainer="FACETPaths", hyperparameters=params)
    model.train(x, y)
    model.prepare()
    preds = model.predict(x)

    # anomaly detection performance
    accuracy, precision, recall, f1 = classification_metrics(preds, y, verbose=True)

    # Q-stastic
    # Q, qs = model.detectors[0].compute_qs(x, y)
    # print("Q-Statistic:", Q)

    # jaccard similarity
    J, jaccards = compute_jaccard(model.detectors[0])
    print("Jaccard Index:", J)

    # generate the explanations
    explanations = model.explain(x, y)

    # measure model performance
    accuracy, precision, recall, f1 = classification_metrics(preds, y, verbose=False)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)
    coverage_ratio = coverage(explanations)
    print("coverage_ratio:", coverage_ratio)
    mean_dist = average_distance(x, explanations, distance_metric="Euclidean")
    print("mean_dist:", mean_dist)
    mean_length = average_distance(x, explanations, distance_metric="FeaturesChanged")
    print("mean_length", mean_length)


if __name__ == "__main__":
    # run_ds = DS_NAMES.copy()
    # run_ds.remove("spambase")
    # compare_methods(['cancer', 'magic', 'vertebral', 'glass'], num_iters=5,
    #                 explainers=["FACETTrees", "BestCandidate", "OCEAN"])
    simple_run("magic")
