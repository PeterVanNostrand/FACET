import numpy as np
from numpy.core.fromnumeric import var
from heead import HEEAD
import matplotlib.pyplot as plt
from utilities.metrics import coverage
from utilities.metrics import classification_metrics
from utilities.metrics import mean_distance
from dataset import load_data
from dataset import all_datasets
from experiments import *


def simple_run(dataset_name):
    # Load the dataset
    x, y = load_data(dataset_name)

    # Create, train, and predict with the model
    model = HEEAD(detectors=["RandomForest"], aggregator="LogisticRegression", explainer="GraphMerge")
    model.train(x, y)
    preds = model.predict(x)

    # anomaly detection performance
    accuracy, precision, recall, f1 = classification_metrics(preds, y, verbose=True)

    # Q-stastic
    # Q, qs = model.detectors[0].compute_qs(x, y)
    # print("Q-Statistic:", Q)

    # jaccard similarity
    J, jaccards = model.detectors[0].compute_jaccard()
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
    mean_dist = mean_distance(x, explanations)
    print("mean_dist:", mean_dist)


if __name__ == "__main__":
    # vary_difference()
    # vary_k()
    # vary_dim(all_datasets, explainer="GraphMerge")
    # vary_ntrees()
    simple_run("thyroid")
