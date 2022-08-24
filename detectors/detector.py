from abc import ABC, abstractmethod


'''
The Detector class is a purely virtual class which represent the base methods which are used to perform anomaly detection, the results of these detectors are combined by the Aggregator. In principle any anomaly detection method which can implement the required functions can be used.
'''


class Detector(ABC):

    '''
    Function to instantiate the detector method being used. Hyparameters should be a dictionary which contains the neccesary configuration values for the method. For example hyperparameters={paramA:0.1, paramB: [1, 2, 3]}
    '''
    @abstractmethod
    def __init__(self, hyperparameters=None):
        pass

    '''
    Function to perform training of the given detector
    x should be an array of shape [nsamples, nfeatures] containing the input samples
    y if provided should be an array of shape [nsamples] containing the true ylabels
    '''
    @abstractmethod
    def train(self, x, y=None):
        pass

    '''
    Function to make a prediction using the given detector
    x should be an array of shape [nsamples, nfeatures] conatining the input samples
    should return y, an array of shape [nsamples] containing the generated labels
    '''
    @abstractmethod
    def predict(self, x):
        pass
