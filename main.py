import numpy as np
from numpy.core.fromnumeric import var
from heead import HEEAD
import matplotlib.pyplot as plt
from utilities.metrics import coverage
from utilities.metrics import classification_metrics
from utilities.metrics import mean_distance
from dataset import load_data
from experiments import *


def simple_run(dataset_name):
    # Load the dataset
    x, y = load_data(dataset_name)

    # Create, train, and predict with the model
    model = HEEAD(detectors=["RandomForest"], aggregator="LogisticRegression", explainer="BestCandidate")
    model.train(x, y)
    preds = model.predict(x)

    # anomaly detection performance
    accuracy, precision, recall, f1 = classification_metrics(preds, y, verbose=True)

    # generate the explanations
    explanations = model.explain(x, y)

    # explanation performance
    print("coverage:", coverage(explanations))
    print("mean distance: ", mean_distance(x, explanations))


if __name__ == "__main__":
    # vary_difference()
    # vary_k()
    vary_dim()
    # vary_ntrees()
    # simple_run("thyroid")
