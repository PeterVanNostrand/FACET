import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as skGradientBoostingClassifier
from detectors.detector import Detector


class GradientBoostingClassifier(Detector):
    def __init__(self, hyperparameters=None, random_state=None):
        self.params = hyperparameters.get("GradientBoostingClassifier")

        # number of trees
        self.loss: int = self.parse_param("gbc_loss", "log_loss")
        self.lr: int = self.parse_param("gbc_learning_rate", 0.1)
        self.ntrees: int = self.parse_param("gbc_ntrees", 100)
        self.maxdepth: int = self.parse_param("gbc_maxdepth", 3)
        self.init_strategy: int = self.parse_param("gbc_init", "zero")

        # create the classifier
        self.model = skGradientBoostingClassifier(
            loss=self.loss, learning_rate=self.lr, n_estimators=self.ntrees, max_depth=self.maxdepth, random_state=random_state, init=self.init_strategy)

    def parse_param(self, param: str, default_val):
        '''
        Check self.params for the given value and return it, if not present return the default value
        '''
        if param in self.params:
            return self.params[param]
        else:
            print("No {} set, using default {}={}".format(param, param, default_val))
            return default_val

    def train(self, x, y=None):
        self.model.fit(x, y)

        # determine the initial value
        if self.init_strategy == "zero":
            self.init_value = 0.0
        else:
            inital_estimator = self.model.init_
            arbitratry_instance = np.zeros(shape=(x.shape[0]))
            class_one_val = inital_estimator.predict_proba(arbitratry_instance)[0, 1]
            self.init_value = class_one_val

    def predict(self, x):
        return self.model.predict(x)

    def apply(self, x):
        return self.model.apply(x)
