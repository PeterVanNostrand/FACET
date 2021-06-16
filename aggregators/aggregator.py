from abc import ABC, abstractmethod


class Aggregator(ABC):

    '''
    Function to instantiate the aggregation method being used. Hyparameters should be a dictionary which contains the neccesary configuration values for the method. For example hyperparameters={paramA:0.1, paramB: [1, 2, 3]}
    '''
    @abstractmethod
    def __init__(self, hyperparameters=None):
        pass

    @abstractmethod
    def train(self, preds, y):
        pass

    @abstractmethod
    def aggregate(self, preds):
        pass

    @abstractmethod
    def get_weights(self):
        pass
