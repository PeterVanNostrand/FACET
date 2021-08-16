from sys import prefix
from detectors.detector import Detector
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
from sklearn import tree
import numpy as np
from utilities.tree_tools import TreeContraster
from utilities.tree_tools import get_best_of_tree

from utilities.metrics import euclidean_distance

import multiprocessing as mp
from functools import partial


class RandomForest(Detector):
    def __init__(self, hyperparameters=None):
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

            # number of trees
            if hyperparameters.get("rf_ntrees") is None:
                print("No rf_ntrees set, using default ntrees=100")
                self.ntrees = 100
            else:
                self.ntrees = hyperparameters.get("rf_ntrees")
        else:
            self.difference = 0.01
            self.distance_fn = euclidean_distance
            self.k = 1
            self.ntrees = 20

        # create the classifier
        self.model = skRandomForestClassifier(n_estimators=self.ntrees)

    def train(self, x, y=None):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_candidate_examples(self, x, y):
        '''
        A function for getting the best `k` contrastive example candidates for the samples of `x`. The *best* examples are those which result in a change of predicted class and have a minimal change from `x[i]`.

        The offset amount used in contrastive example generation and the distance metric used to select the minimal examples are set by the hyperparameters `rf_difference` and `rf_distance` determined during initialization

        Parameters
        ----------
        x               : an array of samples, dimensions (nsamples, nfeatures)
        y               : an array of labels which correspond to the labels, (nsamples, )

        Returns
        -------
        final_examples : an array of contrastive examples with dimensions (nsamples, k, nfeatures). Each of final_examples[i] corresponds to an array of the best k examples which explain x[i]
        '''

        final_examples = np.empty(shape=(x.shape[0], self.k, x.shape[1]))

        trees = self.model.estimators_
        all_examples = [[]] * x.shape[0]  # a list to store the array of examples, one for each example

        # for each tree, get an example for each sample in x
        for t in trees:
            # get an array of candidate examples for each instance in x
            helper = TreeContraster(t)
            tree_examples = helper.construct_examples(x, y, self.difference)

            # merge n_sample arrays from this tree with those from the other trees
            for i in range(x.shape[0]):
                if len(all_examples[i]) == 0:
                    all_examples[i] = tree_examples[i]
                else:
                    if(len(tree_examples[i]) > 0):
                        all_examples[i] = np.vstack([all_examples[i], tree_examples[i]])

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

    def get_candidate_examples_treewise(self, x, y):
        '''
        A function for getting the best contrastive example from each tree for each of the samples of `x`. The *best* examples are those which result in a change of predicted class on that tree and have a minimal change from `x[i]`

        Parameters
        ----------
        x               : an array of samples, dimensions (nsamples, nfeatures)
        y               : an array of labels which correspond to the labels, (nsamples, )

        Returns
        -------
        all_examples : an array of contrastive examples with dimensions (ntrees, nsamples, nfeatures)
        '''

        trees = self.model.estimators_
        num_proc = np.min((self.ntrees, mp.cpu_count()))
        with mp.Pool(num_proc) as p:
            my_func = partial(get_best_of_tree, rf=self, x=x, y=y)
            results = p.map(my_func, trees)

        all_examples = np.stack(results)
        return all_examples

    def get_features_used(self, tree_id):
        '''
        Determines what features are used by a specified tree in the ensemble

        Parameters
        ----------
        tree_id : the index of the tree in the ensemble to be considered

        Returns
        -------
        used_features : an array of integers containing the index of features used by the given tree
        '''
        feat_import = self.model.estimators_[tree_id].feature_importances_
        used_features = np.argwhere(feat_import > 0).T[0]
        return used_features

    def get_tree_information(self):
        '''
        Characterize the forest by computing the average number of nodes, average number of leaves, and average depth of the trees

        Returns
        -------
        avg_nnodes : the average number of nodes in a tree of the forest including interior and leaf notes
        avg_nleaves : the average number of leaf nodes in a tree of the forest
        avg_depth : the average depth of a tree in the forest
        '''
        trees = self.model.estimators_
        sum_nnodes = 0
        sum_nleaves = 0
        sum_depth = 0
        for t in trees:
            sum_nnodes += t.tree_.node_count
            sum_nleaves += t.get_n_leaves()
            sum_depth += t.get_depth()

        avg_nnodes = sum_nnodes / self.ntrees
        avg_nleaves = sum_nleaves / self.ntrees
        avg_depth = sum_depth / self.ntrees

        return avg_nnodes, avg_nleaves, avg_depth

    def compute_qs(self, x, y):
        '''
        A method for computing the Q-Statistics of the Random Forest ensemble

        Parameters
        ----------
        x : a set of data samples
        y : the labels which correspond to the samples in x

        Returns
        -------
        Q : the Q-Statistic for the ensemble
        qs : the q stastic for each pair in the ensemble
        '''

        npairs = (int)((self.ntrees * (self.ntrees - 1)) / 2)
        qs = np.empty(shape=(npairs,))

        trees = self.model.estimators_
        index = 0
        for i in range(self.ntrees):
            for k in range(i+1, self.ntrees):
                qs[index] = self.compute_q_pair(trees[i], trees[k], x, y)
                index += 1

        Q = np.average(qs)
        return Q, qs

    def compute_q_pair(self, t1, t2, data, labels):
        '''
        Compute the average pairwise Q-Statistic, a measure of diversity, for the given trees

        Parameters
        ----------
        t1 : a sklearn decision tree classifier
        t2 : a second sklearn decision tree classifier
        data : an array of data samples with dimensions (nsamples, nfeatures)
        labels : an array of labels with shape (nsamples,) corresponding to the samples in `data`
        '''
        # The Q-Statistic is defined as
        #         (N^11 * N^00) - (N^01 * N^10)
        # Q_i,k = -----------------------------
        #         (N^11 * N^00) + (N^01 * N^10)

        preds1 = t1.predict(data)
        preds2 = t2.predict(data)

        # y_i,j=1 iff detector i correctly recognizes sample j, 0 otherwise
        y1 = (preds1 == labels) * 1
        y2 = (preds2 == labels) * 1

        # Where N^ab is the number of elements where yi=a and yk=b
        n11 = np.logical_and((y1 == 1), (y2 == 1)).sum()
        n00 = np.logical_and((y1 == 0), (y2 == 0)).sum()
        n01 = np.logical_and((y1 == 0), (y2 == 1)).sum()
        n10 = np.logical_and((y1 == 1), (y2 == 0)).sum()

        # TODO: example case where Q-statistic is not defined
        if ((n11*n00) + (n01 * n10)) == 0:
            Qik = 0
        else:
            Qik = ((n11 * n00) - (n01 * n10)) / ((n11*n00) + (n01 * n10))

        return Qik

    def compute_jaccard(self):
        '''
        Computes the pairwise jaccard index for all pairs of trees and returns the forest wide average

        Returns
        -------
        J : the pairwise jaccard index averaged over all pairs of trees in the forest
        jaccards : an array containing the pairwise jaccard index for every combination of two trees in the forest
        '''
        npairs = (int)((self.ntrees * (self.ntrees - 1)) / 2)
        jaccards = np.empty(shape=(npairs,))

        trees = self.model.estimators_
        index = 0
        for i in range(self.ntrees):
            for k in range(i+1, self.ntrees):
                jaccards[index] = self.compute_jaccard_pair(trees[i], trees[k])
                index += 1

        J = np.average(jaccards)
        return J, jaccards

    def compute_jaccard_pair(self, t1, t2):
        '''
        Computes the jaccard similarity of the feature sets used by the given trees

        Parameters
        ----------
        t1 : a sklearn decision tree classifier
        t2 : a second sklearn decision tree classifier

        Returns
        -------
        Jik : The jaccard similarity between the feature sets used by `t1` and `t2`. The Jaccard similarity between two sets is given as `J = |A intersect B| / |A   union   B|`
        '''

        # get the features used be each tree, f[i]>0 iff t uses feature i
        f1 = t1.feature_importances_
        f2 = t2.feature_importances_

        # determine the cardinality of the intersection and union of the feature sets
        a_intersect_b = np.logical_and((f1 > 0), (f2 > 0)).sum()
        a_union_b = np.logical_or((f1 > 0), (f2 > 0)).sum()

        # compute the jaccard index
        Jik = a_intersect_b / a_union_b

        return Jik

    def compute_n_features(self):
        '''
        Computes the mean number of features used by all trees in the forest

        Returns
        -------
        avg_features_used : the average number of features used by the trees of the forest
        '''
        n_features_used = np.empty(shape=(self.ntrees,))

        trees = self.model.estimators_
        for i in range(self.ntrees):
            tree_features = trees[i].feature_importances_
            n_features_used[i] = (tree_features > 0).sum()

        avg_features_used = np.average(n_features_used)
        return avg_features_used

    def get_tree_adjacency(self):
        '''
        Treating the trees in the forest as a graph, with each tree being a node, define an edge as weight 1.0 between two trees if they use a fully disjoint set of features as 0.0 if they use exactly the same set of features

        Returns
        -------
        adjacency : a matrix of shape [ntrees, ntrees] with each element adjacency[i][j] representing the weight of the edge in a graph between nodes i and j. The diagonal of the matrix is set to zero to prevent loops 
        '''
        trees = self.model.estimators_

        # Build matrix for tree subset similarity using jaccard index
        similarity = np.zeros(shape=(self.ntrees, self.ntrees))

        trees = self.model.estimators_
        for i in range(self.ntrees):
            for k in range(i+1, self.ntrees):
                jik = self.compute_jaccard_pair(trees[i], trees[k])
                similarity[i][k] = jik
                similarity[k][i] = jik

        adjacency = (1.0 - similarity)
        np.fill_diagonal(adjacency, 0)  # remove self edges

        return adjacency
