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
