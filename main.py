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

    y[norm_idx] = 1
    y[anom_idx] = -1

    keep_idxs = np.append(norm_idx, anom_idx)

    x = x[keep_idxs]
    y = y[keep_idxs]

    return x, y


if __name__ == "__main__":

    # Load the dataset
    x, y = load_data()

    # Create, train, and predict with the model
    model = HEEAD(config=["IsolationForest", "IsolationForest"], agg="LogisticRegression")
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
