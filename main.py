import numpy as np
from heead import HEEAD


def load_data():
    from sklearn import datasets
    # import the iris dataset for testing
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # iris has three classes, lets use just two to simluate anomaly detection
    norm_class = 1
    anom_class = 2

    norm_idx = np.where(y == 1)[0]
    anom_idx = np.where(y == 2)[0]

    keep_idxs = np.append(norm_idx, anom_idx)

    x = x[keep_idxs]
    y = y[keep_idxs]

    return x, y


if __name__ == "__main__":

    # DATASET
    x, y = load_data()

    model = HEEAD(config=["IsolationForest", "IsolationForest"], agg="LogisticRegression")
    model.train(x, y)
    preds = model.predict(x)
    error = preds - y
    print(error)
