# handle circular imports that result from typehinting
from __future__ import annotations

# core python packages
import math
from os import replace

# scientific and utility packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from tqdm import tqdm
import pandas as pd

# graph packages
import networkx as nx
from networkx.algorithms.approximation import max_clique as get_max_clique

# custom classes
from utilities.metrics import dist_euclidean
from utilities.metrics import dist_features_changed
from explainers.explainer import Explainer
from detectors.random_forest import RandomForest
from explainers.branching import BranchBound
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from heead import HEEAD


class FACETIndex(Explainer):
    def __init__(self, model, hyperparameters: dict = None):
        self.model: HEEAD = model
        self.parse_hyperparameters(hyperparameters)

    def prepare(self, data=None):
        rf_detector: RandomForest = self.model.detectors[0]
        rf_trees = rf_detector.model.estimators_
        self.rf_ntrees = len(rf_trees)
        self.rf_nclasses = rf_detector.model.n_classes_
        self.rf_nfeatures = len(rf_detector.model.feature_importances_)

        # the number of trees which much aggree for a prediction
        self.majority_size = math.floor(self.rf_ntrees / 2) + 1

        self.build_paths(rf_trees)
        self.index_paths(self.rf_nclasses)
        self.index_rectangles(data)

    def index_rectangles(self, xtrain):
        '''
        This method using the training data to enumerate a set of hyper-rectangles of each classes and adds them to an index for later searching during explanation
        '''
        self.initialize_index()
        preds = self.model.predict(xtrain)
        rf_detector: RandomForest = self.model.detectors[0]
        leaves = rf_detector.apply(xtrain)

        # for each instance in the training set
        for instance, class_id, leaf_ids in zip(xtrain, preds, leaves):
            rect = self.enumerate_rectangle(leaf_ids)
            self.add_to_index(class_id, rect)
        a = 2

    def add_to_index(self, class_id, rectangle):
        '''
        Add the rectangle to the index
        '''
        self.index[class_id].append(rectangle)

    def initialize_index(self):
        '''
        Creates an empty index to store the hyper-rectangles. Functionized to allow for multiple indexing options
        '''
        self.index = [[] for _ in range(self.rf_nclasses)]

    def enumerate_rectangle(self, leaf_ids):
        '''
        Given a set of scikitlearns leaf_ids, one per tree in the ensemble, extracts the paths which correspond to these leaves and constructs the corresponding hyper-rectangle
        '''
        feature_bounds = np.tile(np.inf, (self.rf_nfeatures, 2))

        # for each tree, find the indexed path that ends in the selected leaf node and
        # merge that path into the bounds hyper-rectangle
        for tid in range(self.rf_ntrees):
            for pid in range(len(self.all_paths[tid])):
                path_leaf = self.all_paths[tid][pid][-1, 0]  # get the id of the leaf node at the end of the path
                if path_leaf == leaf_ids[tid]:  # if it matches the found leaf
                    # walk the path and update the feature bounds
                    path = self.all_paths[tid][pid]
                    for i in range(path[:-1, :].shape[0]):
                        feature = int(path[i, 1])
                        condition = int(path[i, 2])
                        threshold = path[i, 3]
                        if condition == 0:  # less than or equal to, update max value to be lower
                            feature_bounds[feature][1] = min(threshold, feature_bounds[feature][1])
                        else:  # greater than, update min value to be higher
                            feature_bounds[feature][0] = max(threshold, feature_bounds[feature][0])

        return feature_bounds

    def build_paths(self, trees):
        '''
        Walks each tree and extracts each path from root to leaf into a data structure. Each tree is represented as a list of paths, with each path stored into an array

        Returns
        -------
        all_paths: a list of length ntrees
        '''
        ntrees = len(trees)
        all_paths = [[] for _ in range(ntrees)]
        for i in range(ntrees):
            all_paths[i] = self.__in_order_path(t=trees[i], built_paths=[])

        self.all_paths = all_paths

    def index_paths(self, nclasses):
        '''
        Creates a pair of data structures which index the paths of the decision trees. By iterating by tree and then by path each path is assigned an increasing index. Each class is indexed independently

        treepath_to_idx: takes [tree_id, path_id] and returns [class_id, index]
        idx_to_treepath: takes [class_id, index] and returns [tree_id, path_id]
        '''

        ntrees = len(self.all_paths)

        # takes [tree_id, path_id] and turns it into a [class_id, index]
        treepath_to_idx = []  # list of (ntrees, npaths, 2)

        # takes [class_id, index] and turns it into (tree_id, path_id)
        idx_to_treepath = [[] for _ in range(nclasses)]  # list of (nclasses, npaths)

        indexs = [0] * nclasses
        for i in range(ntrees):
            treei_to_idx = []
            for j in range(len(self.all_paths[i])):
                p = self.all_paths[i][j]
                path_class = int(p[-1, 3])
                treei_to_idx.append((path_class, indexs[path_class]))
                idx_to_treepath[path_class].append((i, j))
                indexs[path_class] += 1
            treepath_to_idx.append(treei_to_idx)

        total_paths = 0
        for idx in indexs:
            total_paths += idx

        if self.verbose:
            print("Num paths: {}".format(total_paths))

        self.total_paths = total_paths
        self.npaths = indexs
        self.treepath_to_idx = treepath_to_idx
        self.idx_to_treepath = idx_to_treepath

    def __in_order_path(self, t, built_paths=[], node_id=0, path=[]):
        '''
        An algorithm for pre-order binary tree traversal. This walks through the entire tree enumerating each paths the root node to a leaf.

        Each path is represented by an array of nodes
        Each node is reprsented by a tuple
            For internal nodes this is
                [node_id, feature, cond, threshold]
            While for leaf nodes this is
                [node_id, -1, -1, class_id]
            Where cond is 0 for (<= )and 1 for (>)
        The tree is reprsented as a list of these arrays

        Parameters
        ----------
        t           : the decision tree classifier to travers
        built_paths : the return values, a list of tuples (`class_id` = integer class ID, `path=[f, g, h, i, j]`)
        node_id     : the `node_id` to start traversal at, defaults to root node of tree (`node_id=0`)
        path        : the starting path up to by not including the node referenced by `node_id`, defaults to empty

        Returns
        -------
        None : see the output parameter `built_paths`
        '''

        # process current node
        feature = t.tree_.feature[node_id]
        if feature >= 0:  # this is an internal node
            threshold = t.tree_.threshold[node_id]

            # process left child, conditioned (<=)
            left_path = path.copy()
            left_path.append([node_id, feature, 0, threshold])
            self.__in_order_path(t=t, built_paths=built_paths, node_id=t.tree_.children_left[node_id], path=left_path)

            # process right node, conditioned (>)
            right_path = path.copy()
            right_path.append([node_id, feature, 1, threshold])
            self.__in_order_path(t=t, built_paths=built_paths, node_id=t.tree_.children_right[node_id], path=right_path)

            return built_paths

        else:  # this is a leaf node
            class_id = np.argmax(t.tree_.value[node_id])
            path = path.copy()
            path.append([node_id, -1, -1, class_id])

            # store the completed path and exit
            finished_path = np.array(path)
            built_paths.append(finished_path)
            return built_paths

    def compute_syth_thresholds(self, path_idxs, nfeatures):
        # the minimum and maximum possible values for each feature, represented by pair [min_value, max_value]
        feature_bounds = [[-float("inf"), float("inf")] for _ in range(nfeatures)]  # min_value < x'[i] < max_value

        for index in path_idxs:
            tree_id = index[0]
            path_id = index[1]
            p = self.all_paths[tree_id][path_id]
            for i in range(p[:-1, :].shape[0]):
                feature = int(p[i, 1])
                condition = int(p[i, 2])
                threshold = p[i, 3]
                if condition == 0:  # less than or equal to, update max value to be lower
                    feature_bounds[feature][1] = min(threshold, feature_bounds[feature][1])
                else:  # greater than, update min value to be higher
                    feature_bounds[feature][0] = max(threshold, feature_bounds[feature][0])

        return feature_bounds

    def fit_thresholds(self, xprime, feature_bounds):
        '''
        For each feature adjust xprime to meet the sythesized rule set
        '''
        for j in range(len(feature_bounds)):
            min_value = feature_bounds[j][0]
            max_value = feature_bounds[j][1]

            if(not xprime[j] > min_value):
                xprime[j] = min((min_value + (min_value * self.offset)), max_value)
            if(not xprime[j] < max_value):
                xprime[j] = max((max_value - (max_value * self.offset)), min_value)

        return xprime

    def explain(self, x: np.ndarray, y: np.ndarray):
        '''
        Parameters
        ----------
        x               : an array of samples, dimensions (nsamples, nfeatures)
        y               : an array of predicted labels which correspond to the labels, (nsamples, )

        Returns
        -------
        xprime : an array of contrastive examples with dimensions (nsamples, nfeatures)

        explains a given instance by growing a clique around the region of xi starting with trees which predict the counterfactual class
        '''
        xprime = x.copy()  # an array for the constructed contrastive examples

        # assumimg binary classification [0, 1] set counterfactual class
        counterfactual_classes = ((y - 1) * -1)

        # check that all counterfactuals result in a different class
        preds = self.model.predict(xprime)
        failed_explanation = (preds == y)
        xprime[failed_explanation] = np.tile(np.inf, x.shape[1])

        if self.verbose:
            print("failed x':", failed_explanation.sum())

        return xprime

    def parse_hyperparameters(self, hyperparameters: dict):
        self.hyperparameters = hyperparameters

        # distance metric for explanation
        if hyperparameters.get("expl_distance") is None:
            print("No expl_distance function set, using Euclidean")
            self.distance_fn = dist_euclidean
        elif hyperparameters.get("expl_distance") == "Euclidean":
            self.distance_fn = dist_euclidean
        elif hyperparameters.get("expl_distance") == "FeaturesChanged":
            self.distance_fn = dist_features_changed
        else:
            print("Unknown expl_distance function {}, using Euclidean distance".format(hyperparameters.get("expl_distance")))
            self.distance_fn = dist_euclidean

        # threshold offest for picking new values
        offset = hyperparameters.get("facet_offset")
        if offset is None:
            print("No facet_offset provided, using 0.01")
            self.offset = 0.01
        else:
            self.offset = offset

        # print messages
        if hyperparameters.get("verbose") is None:
            self.verbose = False
        else:
            self.verbose = hyperparameters.get("verbose")
