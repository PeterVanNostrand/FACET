from abc import ABC, abstractmethod


class Explainer(ABC):
    '''
    The explainer class is a purely virtual class which provides an interface for performing explanation
    on set of examples using an ensemle of aggregated detectors
    '''

    @abstractmethod
    def __init__(self, hyperparameters=None):
        '''
        Function to instantiate the explanation method being used.
        Hyparameters should be a dictionary which contains the neccesary configuration values for the method.
        For example hyperparameters={paramA:0.1, paramB: [1, 2, 3]}
        '''
        pass

    @abstractmethod
    def explain(self, x, y, detectors, aggregator):
        '''
        Function to perform explanation synthesis for the given samples using the provided detectors and aggregator
        '''
        pass
