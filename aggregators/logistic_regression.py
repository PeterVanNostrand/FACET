from aggregators.aggregator import Aggregator
from sklearn.linear_model import LogisticRegression as skLogisticRegression


class LogisticRegression(Aggregator):

    def __init__(self, hyperparameters=None):
        self.model = skLogisticRegression(solver="lbfgs")

    def train(self, preds, y):
        self.model.fit(preds, y)

    def aggregate(self, preds):
        return self.model.predict(preds)

    def get_weights(self):
        return self.model.get_params()
