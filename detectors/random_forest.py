from detector import Detector
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from sklearn import tree
import numpy as np
from utilities.tree_tools import TreeContraster
from utilities.metrics import euclidean_distance


class RandomForest(Detector):
    def __init__(self, hyperparameters=None):
        self.model = skRandomForestClassifier()

    def train(self, x, y=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_candidate_examples(self, x, y, difference=0.01, distance_metric="Euclidean", k=1):
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
        trees = self.model.estimators_
        all_examples = [[]] * x.shape[0]  # a list to store the array of examples, one for each example

        # for each tree, get an example for each sample in x
        for t in trees:
            # get an array of candidate examples for each instance in x
            helper = TreeContraster(t)
            tree_examples = helper.construct_examples(x, y, difference, distance_metric)

            # merge n_sample arrays from this tree with those from the other trees
            for i in range(x.shape[0]):
                if len(all_examples[i]) == 0:
                    all_examples[i] = tree_examples[i]
                else:
                    all_examples[i] = np.vstack([all_examples[i], tree_examples[i]])

        # select the distance function which corresonds to the provided distance metric
        if distance_metric == "Euclidean":
            distance_fn = euclidean_distance
        else:
            print("Unknown distance function {}, using Euclidean distance for random forest".format(distance_metric))
            distance_fn = euclidean_distance

        final_examples = np.empty(shape=(x.shape[0], k, x.shape[1]))

        # for each sample pick the top k best candidates that result in a changed class
        for i in range(x.shape[0]):
            # get the info for that sample
            instance = x[i]
            label = y[i]
            candidate_examples = all_examples[i]

            # get the class predicted by the forest for each candidate
            candidate_preds = self.predict(candidate_examples)

            # keep only candidates that result in a changed prediction
            candidate_examples = candidate_examples[candidate_preds != label]

            # sort the candidates according to the distance function
            candidate_dists = distance_fn(instance, candidate_examples)
            sort_idxs = np.argsort(candidate_dists)
            candidate_examples = candidate_examples[sort_idxs]
            candidate_preds = candidate_preds[sort_idxs]

            # return the top k candidates for x if availible
            if candidate_examples.shape[0] >= k:
                top_k_candidates = candidate_examples[:k]
            # if there are fewer than k candidates, pad with [inf, inf, ... , inf]
            else:
                n_missing = k - candidate_examples.shape[0]
                pad_candidates = np.tile(np.inf, (n_missing, instance.shape[0]))
                top_k_candidates = np.vstack((candidate_examples, pad_candidates))

            final_examples[i] = top_k_candidates

            if k == 1:
                final_examples = final_examples.squeeze()

        return final_examples
