import numpy as np
from heead import HEEAD
import matplotlib.pyplot as plt


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

    return x, y


if __name__ == "__main__":

    # Load the dataset
    x, y = load_data()

    # Create, train, and predict with the model
    model = HEEAD(detectors=["RandomForest"],
                  aggregator="LogisticRegression", explainer="BestCandidate")
    model.train(x, y)
    preds = model.predict(x)

    # Compute performance metrics
    tp = np.where((preds == 1) & (y == 1))[0].shape[0]  # true inliers
    fp = np.where((preds == 1) & (y == -1))[0].shape[0]  # false inliers
    tn = np.where((preds == -1) & (y == -1))[0].shape[0]  # true outliers
    fn = np.where((preds == -1) & (y == 1))[0].shape[0]  # false outlier

    print("tp:", tp)
    print("fp:", fp)
    print("tn:", tn)
    print("fn:", fn)

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print()
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall", recall)
    print("f1:", f1)

    explanations = model.explain(x, y)

    conversion_rate = 1 - (np.isnan(explanations).any(axis=1).sum() / x.shape[0])
    print()
    print("conversion rate:", conversion_rate)
