# handle circular imports that result from typehinting
from __future__ import annotations
from gettext import install

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

        all_paths = self.build_paths(rf_trees)
        self.leaf_rects = self.leaves_to_rects(all_paths)
        self.index_rectangles(data)
        # self.check_indexed_rects()

    def leaves_to_rects(self, all_paths: list[list[np.ndarray]]) -> list[dict]:
        '''
        For each path in the ensemble, identifies the leaf node of that path and builds the corresponding hyper-rectangle

        Parameters
        ----------
        The enumerated paths from build_paths, a ragged list of (ntrees, npaths_i) 

        Returns
        -------
        leaf_rects: a list of dictionarys where leaft_rects[i][j] returns the (leaf_class, rect) corresponding to the path ending in scikit node if j from tree i
        '''
        leaf_rects = [{} for _ in range(self.rf_ntrees)]
        for tid in range(self.rf_ntrees):
            for path in all_paths[tid]:
                # get the scikit node id of the leaf and its class
                leaf_node_id = int(path[-1, 0])
                leaf_class = int(path[-1, -1])
                # build and save the hyper-rectangle that corresponds to this leaf
                rect = self.leaf_rect(path)
                # save the class and the hyper-rectangle
                leaf_rects[tid][leaf_node_id] = (leaf_class, rect)
        return leaf_rects

    def check_indexed_rects(self):
        #! DEBUG tool
        for class_id in range(self.rf_nclasses):
            for rect in self.index[class_id]:
                x = np.zeros(self.rf_nfeatures)
                xprime = self.fit_to_rectangle(x, rect)
                pred = self.model.predict([xprime])
                if pred != class_id:
                    print("Bad Rectangle")

    def index_rectangles(self, xtrain):
        '''
        This method using the training data to enumerate a set of hyper-rectangles of each classes and adds them to an index for later searching during explanation
        '''
        self.initialize_index()
        preds = self.model.predict(xtrain)
        rf_detector: RandomForest = self.model.detectors[0]
        # get the leaves that each sample ends up in
        all_leaves = rf_detector.apply(xtrain)  # shape (nsamples in xtrain, ntrees)

        # for each instance in the training set
        for instance, label, leaf_ids in zip(xtrain, preds, all_leaves):
            rect = self.enumerate_rectangle(leaf_ids, label)
            self.add_to_index(label, rect)

    def add_to_index(self, label: int, rectangle: np.ndarray) -> None:
        '''
        Add the rectangle to the index
        '''
        self.index[label].append(rectangle)

    def initialize_index(self) -> None:
        '''
        Creates an empty index to store the hyper-rectangles. Functionized to allow for multiple indexing options
        '''
        self.index = [[] for _ in range(self.rf_nclasses)]

    def enumerate_rectangle(self, leaf_ids: list[int], label: int) -> np.ndarray:
        '''
        Given a set of scikitlearns leaf_ids, one per tree in the ensemble, extracts the paths which correspond to these leaves and constructs a majority size hyper-rectangle from their interserction

        Paramters
        ---------
        leaf_ids: a list of integer values of size ntrees, where leaf_ids[i] is the scikit learn integer id of the leaf node selected by tree i
        '''
        all_bounds = []
        for tree_id in range(self.rf_ntrees):
            leaf_class, leaf_rect = self.leaf_rects[tree_id][leaf_ids[tree_id]]
            if leaf_class == label:
                all_bounds.append(leaf_rect)
        rect = self.select_intersection(all_bounds, label)
        return rect

    def select_intersection(self, all_bounds: list[np.ndarray], label: int) -> np.ndarray:
        '''
        Given a list of ntrees hyper-rectangles each corresponing to a leaf node, constructs a majority size hyper-rectangle by taking the intersection of nmajority leaf node hyper-rectangles

        Parameters
        ----------
        all_bounds: a list of ntrees numpy arrays, each of size (nfeatures, 2) generated by leaf_rect
        '''
        # initialize the hyper-rectangle as unbounded on both side of all axis
        rect = np.zeros((self.rf_nfeatures, 2))
        rect[:, 0] = -np.inf
        rect[:, 1] = np.inf

        # Simple solution: take the intersection of the first nmajority hyper-rectangles
        i = 0
        while i < len(all_bounds) and i < self.majority_size:
            rect[:, 0] = np.maximum(rect[:, 0], all_bounds[i][:, 0])  # intersction of minimums
            rect[:, 1] = np.minimum(rect[:, 1], all_bounds[i][:, 1])  # intersection of maximums
            i += 1

        # ! debug
        # x = np.zeros(self.rf_nfeatures)
        # xprime = self.fit_to_rectangle(x, rect)
        # pred = int(self.model.predict([xprime]))
        # if pred != label:
        #     print("Bad Rectangle")

        return rect

    def leaf_rect(self, path: np.ndarray) -> np.ndarray:
        '''
        Given a path of a decision tree, compute the hyper-rectangle of size 1 which corresponds to the leaf node of this branch by computing the minimum and maximum threshold values along each axis

        Parameters
        ---------
        path : a numpy array representing the path as constructed by __in_order_path

        Returns
        -------
        feature_bounds : a numpy array of size `(nfeatures, 2)` where `feature_bounds[i][0]` represents the minimum threshold of feature `i` and `feature_bounds[i][1]` the maximum threshold
        '''
        # initialize the thresholds as unbounded on both side of all axis
        feature_bounds = np.zeros((self.rf_nfeatures, 2))
        feature_bounds[:, 0] = -np.inf
        feature_bounds[:, 1] = np.inf

        for i in range(path[:-1, :].shape[0]):
            feature = int(path[i, 1])
            condition = int(path[i, 2])
            threshold = path[i, 3]
            if condition == 0:  # less than or equal to, update max value to be lower
                feature_bounds[feature][1] = min(threshold, feature_bounds[feature][1])
            else:  # greater than, update min value to be higher
                feature_bounds[feature][0] = max(threshold, feature_bounds[feature][0])
        return feature_bounds

    def build_paths(self, trees: list[tree.DecisionTreeClassifier]) -> list[list[np.ndarray]]:
        '''
        Walks each tree and extracts each path from root to leaf into a data structure. Each tree is represented as a list of paths, with each path stored into an array

        Returns
        -------
        all_paths: a list of length ntrees, where all_paths[i] contains a list of the paths in tree i each represented by a numpy array as generated by in_order_path
        '''
        ntrees = len(trees)
        all_paths = [[] for _ in range(ntrees)]
        for i in range(ntrees):
            all_paths[i] = self.__in_order_path(t=trees[i], built_paths=[])

        return all_paths

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

    def is_inside(self, x: np.ndarray, rect: np.ndarray) -> bool:
        '''
        Returns true if x falls inside the given rectangle, false otherwise
        '''
        return (x > rect[:, 0]).all() and (x < rect[:, 1]).all()

    def fit_to_rectangle(self, x: np.ndarray, rect: np.ndarray) -> np.ndarray:
        '''
        Computes xprime, an adjusted copy of x that falls within the bounds of the given rectangle

        Parameters
        ----------
        x: an instance array of shape (nfeatures,)
        rect: a numpy array of shape (nfeatures, 2) representing a hyper-rectangle as constructed by enumrate_rectangle

        Returns
        -------
        xprime: the adjusted instance
        '''
        xprime = x.copy()
        # determine which values need to adjusted to be smaller, and which need to be larger
        low_values = xprime < rect[:, 0]
        high_values = xprime > rect[:, 1]

        # check that the offset will not overstep the min or max value, this occurs when the range between
        # a min and max value is less than the offset size
        overstep_range = (rect[:, 1] - rect[:, 0]) <= self.offset
        # identify features which need to be adjusted and will overstep the min or max value
        idx_overstep = np.logical_and(overstep_range, np.logical_or(low_values, high_values))

        # increase low feature values to one offset above min
        xprime[low_values] = rect[low_values, 0] + self.offset
        # decrease high features one to one offset below min
        xprime[high_values] = rect[high_values, 1] - self.offset
        # for oversteped bounds use the average between min and max
        xprime[idx_overstep] = rect[idx_overstep, 0] + (rect[idx_overstep, 1] - rect[idx_overstep, 0]) / 2

        #! debug
        # if not self.is_inside(xprime, rect):
        #     print("Bad Counterfactual")

        return xprime

    def explain(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

        for i in range(x.shape[0]):
            closest_rect = None
            min_dist = np.inf

            # find the indexed rectangle of the the counterfactual class that tis cloest
            for rect in self.index[counterfactual_classes[i]]:
                test_instance = self.fit_to_rectangle(x[i], rect)
                dist = self.distance_fn(x[i], test_instance)
                if dist < min_dist:
                    min_dist = dist
                    closest_rect = rect

            # generate a counterfactual example which falls within this rectangle
            xprime[i] = self.fit_to_rectangle(x[i], closest_rect)
            #! debug
            # if not self.is_inside(xprime[i], closest_rect):
            #     print("Bad counterfactual 2")

        # check that all counterfactuals result in a different class
        preds = self.model.predict(xprime)
        failed_explanation = (preds == y)
        xprime[failed_explanation] = np.tile(np.inf, x.shape[1])

        if self.verbose:
            print("failed x':", failed_explanation.sum())

        return xprime

    def parse_hyperparameters(self, hyperparameters: dict) -> None:
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
