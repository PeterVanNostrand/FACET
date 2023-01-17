# from typing_extensions import ParamSpec
import random
import numpy as np

# Detector classes
from detectors.random_forest import RandomForest
# Explainer classes
# from explainers import *
from explainers.best_candidate import AFT
from explainers.facet_index import FACETIndex
from explainers.mace import MACE
from explainers.ocean import OCEAN
from explainers.rf_ocse import RFOCSE


class MethodManager():
    def __init__(self, explainer=None, hyperparameters=None, random_state=None):
        self.params = hyperparameters
        self.random_forest = RandomForest(hyperparameters=hyperparameters, random_state=random_state)
        if explainer is not None:
            self.explainer = self.init_explainer(explainer=explainer, hyperparameters=hyperparameters)
        self.random_state = random_state

    def init_explainer(self, explainer, hyperparameters):
        if explainer == "AFT":
            return AFT(manager=self, hyperparameters=hyperparameters)
        elif explainer == "OCEAN":
            return OCEAN(manager=self, hyperparameters=hyperparameters)
        elif explainer == "MACE":
            return MACE(manager=self, hyperparameters=hyperparameters)
        elif explainer == "RFOCSE":
            return RFOCSE(manager=self, hyperparameters=hyperparameters)
        elif explainer == "FACETIndex":
            return FACETIndex(manger=self, hyperparameters=hyperparameters)
        else:
            print("Unknown explainer type of " + explainer)
            print("using FACETIndex")
            return FACETIndex(manger=self, hyperparameters=hyperparameters)

    def set_explainer(self, explainer=None, random_state=None):
        random.seed(random_state)
        np.random.seed(random_state)
        self.explainer = self.init_explainer(explainer=explainer, hyperparameters=self.params)

    def train(self, x, y=None):
        self.random_forest.train(x, y)

    def predict(self, x):
        return self.random_forest.predict(x)

    def prepare(self, xtrain: np.ndarray = None, ytrain: np.ndarray = None):
        self.explainer.prepare(xtrain, ytrain)

    def explain(self, x: np.ndarray, y: np.ndarray, k: int = 1, constraints: np.ndarray = None,
                weights: np.ndarray = None, max_dist: float = np.inf, opt_robust=False, min_robust: float = None) -> np.ndarray:
        return self.explainer.explain(x=x, y=y, k=k, constraints=constraints, weights=weights, max_dist=max_dist, opt_robust=opt_robust, min_robust=min_robust)
