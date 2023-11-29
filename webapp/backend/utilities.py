from numpy import ndarray
def normalize(data:ndarray,min_values:ndarray,max_values:ndarray) -> ndarray:
    '''
    Normalizes 1D array of values
    Parameters:
        - data: ndarray of unnormalized values
        - min_values: ndarray of the minimum values for each feature in a dataset
        - max_values: ndarray of the maximum values for each feature in a dataset
    Returns:
        ndarray of normalized values
    '''
    return (data - min_values) / (max_values - min_values)
