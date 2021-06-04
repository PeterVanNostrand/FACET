from detector import Detector
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from sklearn import tree


class RandomForest(Detector):
    def __init__(self, hyperparameters=None):
        self.model = skRandomForestClassifier()

    def train(self, x, y=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_candidate_examples(self, x, y):
        trees = self.model.estimators_
        for t in trees:
            print(tree.plot_tree(t))
            break
