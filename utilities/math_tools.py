import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    '''
    Returns the sigmoid logistic function for each value in z

    Parameters
    ----------
    z: a numpy array of floats

    Returns
    -------
    sigmoid(z), a numpy array of floats
    '''
    return 1 / (1 + np.exp(-z))
