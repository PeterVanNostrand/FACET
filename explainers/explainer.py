from abc import ABC, abstractmethod


class Explainer(ABC):
    '''
    The explainer class is a purely virtual class which provides an interface for performing explanation
    on set of examples using an ensemle of aggregated detectors
    '''

    @abstractmethod
    def __init__(self, model, hyperparameters=None):
        '''
        Function to instantiate the explanation method being used.
        Hyparameters should be a dictionary which contains the neccesary configuration values for the method.
        For example hyperparameters={paramA:0.1, paramB: [1, 2, 3]}
        '''
        pass

    @abstractmethod
    def prepare(self, data=None):
        '''
        Function to initialize the explainer, called after the model is trained.
        '''
        pass

    @abstractmethod
    def explain(self, x, y):
        '''
        Function to perform explanation synthesis for the given samples using the provided detectors and aggregator
        '''
        pass
