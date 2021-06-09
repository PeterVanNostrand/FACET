import numpy as np
from heead import HEEAD
import matplotlib.pyplot as plt
from utilities.metrics import conversion_rate
from utilities.metrics import confusion_matrix
from utilities.metrics import mean_distance


def load_data():
    from sklearn import datasets
    # import the iris dataset for testing
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # iris has three classes, lets use just two to simluate anomaly detection
    norm_class = 1
    anom_class = 2

    norm_idx = np.where(y == norm_class)[0]
    anom_idx = np.where(y == anom_class)[0]

    y[norm_idx] = 1
    y[anom_idx] = -1

    keep_idxs = np.append(norm_idx, anom_idx)

    x = x[keep_idxs]
    y = y[keep_idxs]

    x = x[:80]
    y = y[:80]

    # normalize x on [0, 1]
    max_value = np.max(x, axis=0)
    min_value = np.min(x, axis=0)
    x = (x - min_value) / (max_value - min_value)

    return x, y


if __name__ == "__main__":

    # Load the dataset
    x, y = load_data()

    # Create, train, and predict with the model
    model = HEEAD(detectors=["RandomForest", "RandomForest"],
                  aggregator="LogisticRegression", explainer="BestCandidate")
    model.train(x, y)
    preds = model.predict(x)

    # anomaly detection performance
    print(confusion_matrix(preds, y))

    # generate the explanations
    explanations = model.explain(x, y)

    # explanation performance
    print("conversion rate:", conversion_rate(explanations))
    print("mean distance: ", mean_distance(x, explanations))
