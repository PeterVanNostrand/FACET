import numpy as np
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from scipy.stats import mode

from detectors.detector import Detector


class RandomForest(Detector):
    def __init__(self, hyperparameters=None, random_state=None):
        params = hyperparameters.get("RandomForest")
        # max depth
        if params.get("rf_maxdepth") is None:
            self.maxdepth = None
        else:
            self.maxdepth = params.get("rf_maxdepth")

        # number of trees
        if params.get("rf_ntrees") is None:
            print("No rf_ntrees set, using default ntrees=100")
            self.ntrees = 100
        else:
            self.ntrees = params.get("rf_ntrees")

        # hard voting
        if params.get("rf_hardvoting") is None:
            print("No rf_hardvoting set, using default True")
            self.hard_voting = True
        else:
            self.hard_voting = params.get("rf_hardvoting")

        # create the classifier
        self.model = skRandomForestClassifier(
            n_estimators=self.ntrees, max_depth=self.maxdepth, random_state=random_state)

    def train(self, x, y=None):
        self.model.fit(x, y)

    def predict(self, x):
        if(self.hard_voting):
            tree_preds = np.empty(shape=(self.ntrees, len(x)))
            for i in range(self.ntrees):
                tree_preds[i] = self.model.estimators_[i].predict(x)
            preds = mode(tree_preds, axis=0)[0][0].astype(int)
            return preds
        else:
            return self.model.predict(x)

    def apply(self, x):
        return self.model.apply(x)
