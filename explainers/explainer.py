from abc import ABC, abstractmethod
import numpy as np


class Explainer(ABC):
    '''
    The explainer class is a purely virtual class which provides an interface for performing explanation
    on set of examples using an ensemle of aggregated detectors
    '''

    @abstractmethod
    def __init__(self, manager, hyperparameters: dict = None) -> None:
        '''
        Function to instantiate the explanation method being used.
        Hyparameters should be a dictionary which contains the neccesary configuration values for the method.
        For example hyperparameters={paramA:0.1, paramB: [1, 2, 3]}
        '''
        pass

    @abstractmethod
    def prepare_dataset(self, x: np.ndarray, y: np.ndarray, ds_info) -> None:
        '''
        Function to ready and dataset statistics needed for comparison methods. Method is not to store any samples for use in explanation
        '''
        pass

    @abstractmethod
    def prepare(self, xtrain: np.ndarray = None, ytrain: np.ndarray = None) -> None:
        '''
        Function to initialize the explainer, called after the model is trained.
        '''
        pass

    @abstractmethod
    def explain(self, x: np.ndarray, y: np.ndarray, k: int = 1, constraints: np.ndarray = None, weights: np.ndarray = None, max_dist: float = np.inf, min_robust: float = None, opt_robust: bool = False) -> np.ndarray:
        '''
        Function to perform explanation synthesis for the given samples using the provided detectors and aggregator
        '''
        pass
