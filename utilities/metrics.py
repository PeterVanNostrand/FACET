import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def dist_euclidean(x, xprime, weights=None):
    '''
    Computes the weighted euclidean distance between `x` and `xprime` sqrt(sum(w*(x_i - xprime_i)**2))

    Parameters
    ----------
    x      : a array of dimension d
    xprime : an array of dimension d+1
    weights: an array of shape (dimension,)

    Returns
    -------
    distance : an array of dimension d+1 containing the Euclidean distance between x and each example in xprime
    '''
    diff = x - xprime

    if weights is not None:  # scale the difference by the weights, zero weighted features are unchangeable feature
        diff[np.where(np.logical_and(diff != 0, weights == 0))] = np.inf  # changing unchangeable feature i->dist[i]=inf
        np.divide(diff, weights, out=diff, where=(weights != 0))  # handle div by 0, values are set to 0 or inf

    squared_diff = np.square(diff)
    if len(xprime.shape) == 3:
        sos_diff = np.sum(squared_diff, axis=2)
    elif len(xprime.shape) == 2:
        sos_diff = np.sum(squared_diff, axis=1)
    else:
        sos_diff = np.sum(squared_diff)

    distance = np.sqrt(sos_diff)
    return distance


def dist_manhattan(x, xprime, weights=None):
    '''
    Computes the weighted manhattan distance between `x` and `xprime` sqrt(sum(w*(x_i - xprime_i)**2))

    Parameters
    ----------
    x      : a array of dimension d
    xprime : an array of dimension d+1
    weights: an array of shape (dimension,)

    Returns
    -------
    distance : an array of dimension d+1 containing the Euclidean distance between x and each example in xprime
    '''
    diff = np.abs(x - xprime)

    if weights is not None:  # scale the difference by the weights, zero weighted features are unchangeable feature
        diff[np.where(np.logical_and(diff != 0, weights == 0))] = np.inf  # changing unchangeable feature i->dist[i]=inf
        np.divide(diff, weights, out=diff, where=(weights != 0))  # handle div by 0, values are set to 0 or inf

    if len(xprime.shape) == 3:
        distance = np.sum(diff, axis=2)
    elif len(xprime.shape) == 2:
        distance = np.sum(diff, axis=1)
    else:
        distance = np.sum(diff)

    return distance

# def weighted_dist(x, xprime, weights):
#     diff = (x - xprime)
#     squared_diff = np.square(diff)
#     scaled_squared_diff = squared_diff / np.square(weights)


def dist_features_changed(x, xprime):
    '''
    Computes the euclidean distance between `x` and `xprime`

    Parameters
    ----------
    x      : a array of dimension d
    xprime : an array of dimension d+1

    Returns
    -------
    distance : an array of dimension d+1 containing the number of features changed between x and each example in xprime
    '''
    if len(xprime.shape) == 3:
        distance = (x != xprime).sum(axis=2)
    elif len(xprime.shape) == 2:
        distance = (x != xprime).sum(axis=1)
    else:
        distance = (x != xprime).sum()

    return distance


def percent_valid(examples):
    '''
    Returns the ratio of samples for which a valid contrastive example could be found. Note that explainers are responsible for returning [np.inf, ..., np.inf] for any instances which they could not generate a counterfactual for
    '''
    # if a contrastive example could not be found an array of [inf, inf, ... , inf] is returned
    return 1 - ((examples == np.inf).any(axis=1).sum() / examples.shape[0])


def classification_metrics(preds, y, verbose=True):
    out_string = ""

    # Compute performance metrics
    # tp = np.sum((preds == 1) & (y == 1))  # true outliers
    # fp = np.sum((preds == 1) & (y == 0))  # false outliers
    # tn = np.sum((preds == 0) & (y == 0))  # true inliers
    # fn = np.sum((preds == 0) & (y == 1))  # false inliers

    # accuracy = (tp + tn) / (tp + fp + tn + fn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 * ((precision * recall) / (precision + recall))

    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)

    # confusion_matrix = ("| act \ pred | Norm | Anom |\n" +
    #                     "|------------|------|------|\n" +
    #                     "|       Norm | {tp:4d} | {fn:4d} |\n" +
    #                     "|       Anom | {fp:4d} | {tn:4d} |\n").format(tp=tp, fn=fn, fp=fp, tn=tn)
    confusion_matrix = ""

    stastics = ("|  Metric   | Value  |\n" +
                "|-----------|--------|\n" +
                "| accuracy  | {a:0.4f} |\n" +
                "| precision | {p:0.4f} |\n" +
                "| recall    | {r:0.4f} |\n" +
                "| f1        | {f:0.4f} |\n").format(a=accuracy, p=precision, r=recall, f=f1)
    out_string = confusion_matrix + "\n" + stastics

    if verbose:
        print(out_string)

    return accuracy, precision, recall, f1


def average_distance(x, xprime, distance_metric="Euclidean"):
    # select the distance function which corresonds to the provided distance metric
    if distance_metric == "Euclidean":
        distance_fn = dist_euclidean
    elif distance_metric == "Manhattan":
        distance_fn = dist_manhattan
    elif distance_metric == "FeaturesChanged":
        distance_fn = dist_features_changed
    else:
        print("Unknown distance function {}, using Euclidean distance for average distance".format(distance_metric))
        distance_fn = dist_euclidean

    # exclude cases where no example was found
    idx_bad_examples = (xprime == np.inf).any(axis=1)
    x = x[~idx_bad_examples]
    xprime = xprime[~idx_bad_examples]

    if xprime.shape[0] == 0:
        # print("No valid examples found")
        avg_dists = np.nan
    else:
        dists = distance_fn(x, xprime)
        avg_dists = np.average(dists)

    return avg_dists
