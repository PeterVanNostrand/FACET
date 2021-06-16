from detectors.detector import Detector
from sklearn.ensemble import IsolationForest as skIsolationForest


class IsolationForest(Detector):
    def __init__(self, hyperparameters=None):
        self.model = skIsolationForest()

    def train(self, x, y=None):
        self.model.fit(x)

    def predict(self, x):
        return self.model.predict(x)

    def get_candidate_examples(self, x, y):
        pass
