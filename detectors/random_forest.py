from sys import prefix
from detector import Detector
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from sklearn import tree
import numpy as np
from utilities.tree_tools import TreeContraster
from utilities.metrics import euclidean_distance
from tqdm import tqdm


class RandomForest(Detector):
    def __init__(self, hyperparameters=None):
        self.model = skRandomForestClassifier()
        if hyperparameters is not None:
            # difference value for explanation
            if hyperparameters.get("rf_difference") is None:
                print("No rf_difference value set, using default 0.01")
                self.difference = 0.01
            else:
                self.difference = hyperparameters.get("rf_difference")

            # distance metric for explanation
            if hyperparameters.get("rf_distance") is None:
                print("No rf_distance function set, using Euclidean")
                self.distance_fn = euclidean_distance
            elif hyperparameters.get("rf_distance") == "Euclidean":
                self.distance_fn = euclidean_distance
            else:
                print("Unknown rf_distance function {}, using Euclidean distance".format(hyperparameters.get("rf_distance")))
                self.distance_fn = euclidean_distance

            # k for explanation
            if hyperparameters.get("rf_k") is None:
                print("No rf_k set, using default k=1")
                self.k = 1
            else:
                self.k = hyperparameters.get("rf_k")
        else:
            self.difference = 0.01
            self.distance_fn = euclidean_distance
            self.k = 1

    def train(self, x, y=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_candidate_examples(self, x, y):
        '''
        A function for getting the best `k` contrastive example candidates for the samples of `x`. The *best* examples are those which result in
        a change of predicted class and have a minimal change from `x[i]`

        Parameters
        ----------
        x               : an array of samples, dimensions (nsamples, nfeatures)
        y               : an array of labels which correspond to the labels, (nsamples, )
        difference      : the percent change relative of the threshold value to offest to each feature to "nudge" the contrastive example past thresholds
        distance_metric : the name of a distance metric to use for sorting the candidates, see `utilities.metrics` for options

        Returns
        -------
        final_examples : an array of contrastive examples with dimensions (nsamples, k, nfeatures). Each of final_examples[i] corresponds to 
        an array of the best k examples which explain x[i]
        '''

        final_examples = np.empty(shape=(x.shape[0], self.k, x.shape[1]))

        trees = self.model.estimators_
        all_examples = [[]] * x.shape[0]  # a list to store the array of examples, one for each example

        # for each tree, get an example for each sample in x
        for t in tqdm(trees, desc="Walking RF Trees"):
            # get an array of candidate examples for each instance in x
            helper = TreeContraster(t)
            tree_examples = helper.construct_examples(x, y, self.difference)

            # merge n_sample arrays from this tree with those from the other trees
            for i in range(x.shape[0]):
                if len(all_examples[i]) == 0:
                    all_examples[i] = tree_examples[i]
                else:
                    all_examples[i] = np.vstack([all_examples[i], tree_examples[i]])

        # for each sample pick the top k best candidates that result in a changed class
        for i in tqdm(range(x.shape[0]), desc="Finding best k DT candiates"):
            # get the info for that sample
            instance = x[i]
            label = y[i]
            candidate_examples = all_examples[i]

            # get the class predicted by the forest for each candidate
            candidate_preds = self.predict(candidate_examples)

            # keep only candidates that result in a changed prediction
            candidate_examples = candidate_examples[candidate_preds != label]

            # sort the candidates according to the distance function
            candidate_dists = self.distance_fn(instance, candidate_examples)
            sort_idxs = np.argsort(candidate_dists)
            candidate_examples = candidate_examples[sort_idxs]
            candidate_preds = candidate_preds[sort_idxs]

            # return the top k candidates for x if availible
            if candidate_examples.shape[0] >= self.k:
                top_k_candidates = candidate_examples[:self.k]
            # if there are fewer than k candidates, pad with [inf, inf, ... , inf]
            else:
                n_missing = self.k - candidate_examples.shape[0]
                pad_candidates = np.tile(np.inf, (n_missing, instance.shape[0]))
                top_k_candidates = np.vstack((candidate_examples, pad_candidates))

            final_examples[i] = top_k_candidates

            if self.k == 1:
                final_examples = final_examples.squeeze()

        return final_examples
