import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from detectors.isolation_forest import IsolationForest
from aggregators.logistic_regression import LogisticRegression


class HEEAD():
    def __init__(self, config=None, agg=None):
        self.detectors = []
        for detector_type in config:
            if detector_type == "IsolationForest":
                d = IsolationForest()
                self.detectors.append(d)
            else:
                print("Unknown detector type of " + detector_type)
                continue
        self.ndetectors = len(self.detectors)

        if agg == "LogisticRegression":
            self.aggregator = LogisticRegression()
        else:
            print("Unknown aggregator type of " + agg)
            print("using logistic regression")
            self.aggregator = LogisticRegression()

    def train(self, x, y=None):
        # Use semi-supervised approach to train aggregator using 5% of data, use remainder as unsupervised for detectors
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33)

        self.__train_detectors(xtrain, ytrain)
        self.__train_aggregator(xtest, ytest)

    def predict(self, x):
        # make predictions using detectors
        preds = None
        for i in range(self.ndetectors):
            pd = self.detectors[i].predict(x)
            if preds is None:
                preds = pd
            else:
                preds = np.stack((preds, pd), axis=1)

        # aggregate the results
        final_preds = self.aggregator.aggregate(preds)
        return final_preds

    def __train_detectors(self, x, y):
        # split data evenly among detectors and train them
        skf = StratifiedKFold(n_splits=self.ndetectors, shuffle=True)
        i = 0
        for train_index, test_index in skf.split(x, y):
            xd = x[train_index]
            yd = y[test_index]
            self.detectors[i].train(x)
            i = i + 1

    def __train_aggregator(self, x, y):
        # make predictions using detectors
        preds = None
        for i in range(self.ndetectors):
            pd = self.detectors[i].predict(x)
            if preds is None:
                preds = pd
            else:
                preds = np.stack((preds, pd), axis=1)

        # perform the aggregator training
        self.aggregator.train(preds, y)
