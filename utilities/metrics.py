import numpy as np


def euclidean_distance(x, xprime):
    '''
    Computes the euclidean distance between `x` and `xprime`

    Parameters
    ----------
    x      : a array of dimension d
    xprime : an array of dimension d+1

    Returns
    -------
    distance : an array of dimension d+1 containing the Euclidean distance between x and each example in xprime
    '''
    diff = x - xprime
    squared_diff = np.square(diff)
    if len(xprime.shape) == 3:
        sos_diff = np.sum(squared_diff, axis=2)
    elif len(xprime.shape) == 2:
        sos_diff = np.sum(squared_diff, axis=1)
    else:
        sos_diff = np.sum(squared_diff)

    distance = np.sqrt(sos_diff)
    return distance


def conversion_rate(examples):
    '''
    Returns the ratio of samples for which a valid contrastive example could be found
    '''
    # if a contrastive example could not be found an array of [inf, inf, ... , inf] is returned
    return 1 - ((examples == np.inf).any(axis=1).sum() / examples.shape[0])


def confusion_matrix(preds, y):
    out_string = ""

    # Compute performance metrics
    tp = np.where((preds == 1) & (y == 1))[0].shape[0]  # true inliers
    fp = np.where((preds == 1) & (y == -1))[0].shape[0]  # false inliers
    tn = np.where((preds == -1) & (y == -1))[0].shape[0]  # true outliers
    fn = np.where((preds == -1) & (y == 1))[0].shape[0]  # false outlier

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    confusion_matrix = ("| act \ pred | Norm | Anom |\n" +
                        "|------------|------|------|\n" +
                        "|       Norm | {tp:4d} | {fn:4d} |\n" +
                        "|       Anom | {fp:4d} | {tn:4d} |\n").format(tp=tp, fn=fn, fp=fp, tn=tn)

    stastics = ("|  Metric   | Value  |\n" +
                "|-----------|--------|\n" +
                "| accuracy  | {a:0.4f} |\n" +
                "| precision | {p:0.4f} |\n" +
                "| recall    | {r:0.4f} |\n" +
                "| f1        | {f:0.4f} |\n").format(a=accuracy, p=precision, r=recall, f=f1)
    out_string = confusion_matrix + "\n" + stastics
    return out_string


def mean_distance(x, xprime, distance_metric="Euclidean"):
    # select the distance function which corresonds to the provided distance metric
    if distance_metric == "Euclidean":
        distance_fn = euclidean_distance
    else:
        print("Unknown distance function {}, using Euclidean distance for mean distance".format(distance_metric))
        distance_fn = euclidean_distance

    # exclude cases where no example was found
    idx_bad_examples = (xprime == np.inf).any(axis=1)
    x = x[~idx_bad_examples]
    xprime = xprime[~idx_bad_examples]

    dists = distance_fn(x, xprime)
    return np.average(dists)
