from detector import Explainer
from sklearn.ensemble import IsolationForest as skIsolationForest


class BestCandidate(Explainer):
    def __init__(self, hyperparameters=None):
        self.model = skIsolationForest()

    def __init__(self, hyperparameters=None):
        pass

    def explain(self, x, detectors, aggregator):
        for i in range(len(detectors)):
            detectors[i].get_candidate_examples(x, y=1)
