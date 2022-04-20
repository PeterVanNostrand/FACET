# handle circular imports that result from typehinting
from __future__ import annotations
from gettext import install

# core python packages
import math
from os import replace
from xmlrpc.client import Boolean

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
from explainers.branching import BranchIndex
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

        # walk each path and store their representation in an array
        self.all_paths = self.build_paths(rf_trees)
        # convert each leaf node into its corresponding hyper-rectangle
        self.leaf_rects = self.leaves_to_rects(self.all_paths)

        # build the index of majority size hyper-rectangles
        if self.enumeration_type == "PointBased":
            self.point_enumerate(data)
        elif self.enumeration_type == "GraphBased":
            self.graph_enumerate(training_data=data)

    def compute_supports(self, data: np.ndarray):
        '''
        Computes the support for each vertex and each pairs of vertices in the classification output of the training data. That is fraction of all samples which are classificed into vertex Vi or pair of vertices Vi, Vj. These values are stored in self.vertex_support[i] and self.pairwise_support[i][j] respectively
        '''
        rf_detector: RandomForest = self.model.detectors[0]
        all_leaves = rf_detector.apply(data)  # leaves that each sample ends up in (nsamples in xtrain, ntrees)

        # compute the support of each vertex in the predictions of the training data
        # i.e. what fraction of the training predictions classify into each leaf
        vertex_supports = []
        pairwise_supports = []
        for class_id in range(self.rf_nclasses):
            # instantiate arrays to hold the support values
            nvertices = self.adjacencys[class_id].shape[0]
            vertex_supports.append(np.zeros(shape=(nvertices,)))
            pairwise_supports.append(np.zeros(shape=(nvertices, nvertices)))

            # for each vertex
            for vertex_i in range(nvertices):
                # get the id of the tree and leaf which correspond to this vertex
                vi_tree_id, vi_leaf_id = self.idx_to_treepath[class_id][vertex_i]
                # find and count all instances where the given tree classified to the given leaf
                contains_vi = all_leaves[:, vi_tree_id] == vi_leaf_id
                vertex_supports[class_id][vertex_i] = contains_vi.sum()
                # compute its pairwise support with other vertices
                for vertex_j in range(vertex_i+1, nvertices):
                    # get the id of the tree and leaf which correspond to this vertex
                    vj_tree_id, leaf_id_j = self.idx_to_treepath[class_id][vertex_j]
                    # find all instances where the given tree classified to the given leaf
                    contains_vj = all_leaves[:, vj_tree_id] == leaf_id_j
                    # find and count all instance where both vi and vj appear
                    pair_support = np.logical_and(contains_vi, contains_vj).sum()
                    # save the pairwise support
                    pairwise_supports[class_id][vertex_i][vertex_j] = pair_support
                    pairwise_supports[class_id][vertex_j][vertex_i] = pair_support
            # support =  support count / num of records (# of training samples)
            vertex_supports[class_id] = vertex_supports[class_id] / data.shape[0]
            pairwise_supports[class_id] = pairwise_supports[class_id] / data.shape[0]

        self.vertex_support = vertex_supports
        self.pairwise_support = pairwise_supports

    def graph_enumerate(self, training_data: np.ndarray) -> None:
        '''
        Converts the random forest ensemble into a graph representation, then uses a branching technique to find k-sized cliques on this graph which correspond to majority sized hyper-rectangles which are added to the index
        '''
        # 1. build the graphs, one per class
        rf_detector: RandomForest = self.model.detectors[0]
        rf_trees: list[tree.DecisionTreeClassifier] = rf_detector.model.estimators_
        self.index_paths(self.rf_nclasses)
        self.find_synthesizeable_paths(rf_trees)
        self.build_graphs(rf_trees, self.rf_nclasses)
        self.compute_supports(data=training_data)

        # 4. explore graph by branching using the support values as priority
        self.initialize_index()
        self.solution_cliques = [[] for _ in range(self.rf_nclasses)]
        for class_id in range(self.rf_nclasses):
            brancher = BranchIndex(
                explainer=self, graph=self.graphs[class_id], class_id=class_id, hyperparameters=self.hyperparameters)
            cliques, rects = brancher.solve()
            self.index[class_id] = rects
            self.solution_cliques[class_id] = cliques

        self.check_indexed_rects()

    def point_enumerate(self, data: np.ndarray) -> None:
        '''
        Uses the provided set of data to select a set of points in the input space, then selects and indexes one hyper-rectangle around each point
        '''
        rect_points = self.select_points(training_data=data)
        self.index_rectangles(data=rect_points)

    def select_points(self, training_data: np.ndarray) -> np.ndarray:
        '''
        Given the training data, selects a set of points which will be used to determine
        the location of each hyper-rectangle to be indexed
        '''
        if self.sample_type == "Training":
            rect_points = training_data
        elif self.sample_type == "Random":
            all_same_class = True
            while all_same_class:
                # create a set of points randomly placed with a uniform distribution along each axis
                rect_points = np.random.uniform(low=0.0, high=1.0, size=(self.n_rects, training_data.shape[1]))
                # check to make sure the resulting set has points of both classes, redraw if needed
                preds = self.model.predict(rect_points)
                all_same_class = len(np.unique(preds)) < 2
        elif self.sample_type == "Augment":
            all_same_class = True
            while all_same_class:
                # take a bootstrap sample of the training data
                rand_idxs = np.random.randint(low=0, high=training_data.shape[0], size=self.n_rects)
                rect_points = training_data[rand_idxs]
                # augment these points with random normally distributed noise
                noise = np.random.normal(loc=0.0, scale=0.5, size=rect_points.shape)
                rect_points += noise
                # check to make sure the resulting set has points of both classes, redraw if needed
                preds = self.model.predict(rect_points)
                all_same_class = len(np.unique(preds)) < 2
        return rect_points

    def explore_index(self, nfeatures: int = 0, npoints: int = 1000, points: np.ndarray = None) -> float:
        '''
        Randomly samples a number of points and checks what percentage of those points fall in the index
        '''
        # generate points if not provided
        if points is None:
            if nfeatures == 0:
                nfeatures = self.rf_nfeatures
            points = np.random.uniform(low=0.0, high=1.0, size=(npoints, nfeatures))
        # check the points
        num_inside = 0
        for p in points:
            if self.inside_index(p):
                num_inside += 1
        percent_covered = num_inside / points.shape[0]
        return percent_covered

    def mean_rect_sizes(self):
        mean_size = np.zeros(shape=(self.rf_nclasses, self.rf_nfeatures))
        # mean_noninf_size = np.zeros(shape=(self.rf_nclasses, self.rf_nfeatures))
        for label in range(self.rf_nclasses):
            for rect in self.index[label]:
                widths = self.rect_width(rect)
                mean_size[label] += widths
            mean_size[label] = mean_size[label] / len(self.index[label])
        return mean_size

    def rect_width(self, rect: np.ndarray) -> np.ndarray:
        '''
        Returns the width of the rectangle along each axis
        Unbounded axis minimums and maximums are treated as zero and one respectively
        '''
        test_rect = rect.copy()
        test_rect[:, 0][test_rect[:, 0] == -np.inf] = 0
        test_rect[:, 1][test_rect[:, 1] == np.inf] = 1
        widths = test_rect[:, 1] - test_rect[:, 0]
        return widths

    def inside_index(self, point: np.ndarray) -> Boolean:
        '''
        Checks if the given point falls within a hyper-rectangle included in the index
        '''
        label = self.model.predict([point])
        covered = False
        i = 0
        while i < len(self.index[label]) and not covered:
            covered = covered or self.is_inside(point, self.index[label][i])
            i += 1
        return covered

    def leaves_to_rects(self, all_paths: list[list[np.ndarray]]) -> list[dict]:
        '''
        For each path in the ensemble, identifies the leaf node of that path and builds the corresponding hyper-rectangle

        Parameters
        ----------
        The enumerated paths from build_paths, a ragged list of (ntrees, npaths_i) 

        Returns
        -------
        leaf_rects: a list of dictionarys where leaft_rects[i][j] returns the (leaf_class, rect) corresponding to the path ending in scikit node id j from tree i
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
                pred = int(self.model.predict([xprime]))
                if pred != class_id:
                    print("Bad Rectangle")

    def index_rectangles(self, data: np.ndarray):
        '''
        This method using the training data to enumerate a set of hyper-rectangles of each classes and adds them to an index for later searching during explanation
        '''
        self.initialize_index()
        preds = self.model.predict(data)
        rf_detector: RandomForest = self.model.detectors[0]
        # get the leaves that each sample ends up in
        all_leaves = rf_detector.apply(data)  # shape (nsamples in xtrain, ntrees)

        # for each instance in the training set
        for instance, label, leaf_ids in zip(data, preds, all_leaves):
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
        Given a set of scikitlearns leaf_ids, one per tree in the ensemble, extracts the paths which correspond to these leaves and constructs a majority size hyper-rectangle from their intersection

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
        # compute how many axes each rectangle bounds, this is equal to the number of noninfinite threshold
        # values in the hyper-rectangle array. Ranges [0, 2*ndims]
        n_bounded_axes = [0] * len(all_bounds)
        mean_rect_widths = [0] * len(all_bounds)
        for i in range(len(all_bounds)):
            n_bounded_axes[i] = np.isfinite(all_bounds[i]).sum()
            mean_rect_widths[i] = self.rect_width(all_bounds[i]).mean()
        order_axes = np.argsort(n_bounded_axes)  # take rects with the fewest bounds first
        order_size = np.argsort(mean_rect_widths)  # take largest rects first
        order_forest = list(range(len(all_bounds)))  # take rects in order of trees in ensemble
        order = order_size

        # initialize the hyper-rectangle as unbounded on both side of all axes
        rect = np.zeros((self.rf_nfeatures, 2))
        rect[:, 0] = -np.inf
        rect[:, 1] = np.inf

        # Simple solution: take the intersection of the first nmajority hyper-rectangles
        i = 0
        while i < len(all_bounds) and i < self.majority_size:
            rect[:, 0] = np.maximum(rect[:, 0], all_bounds[order[i]][:, 0])  # intersection of minimums
            rect[:, 1] = np.minimum(rect[:, 1], all_bounds[order[i]][:, 1])  # intersection of maximums
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

    def find_synthesizeable_paths(self, trees):
        ntrees = len(trees)
        sythesizable_paths = [[[] for _ in range(ntrees)] for _ in range(ntrees)]
        for i in range(ntrees):
            for k in range(ntrees):
                if(i != k):
                    t1_t2_merges = self.get_merges(t1_id=i, t2_id=k)
                    sythesizable_paths[i][k] = t1_t2_merges

        self.sythesizable_paths = sythesizable_paths

    def get_merges(self, t1_id, t2_id):
        '''
        For each path p in t1, identifies all paths in t2 which are mergable with p

        Returns
        ----------
        t1_merges: a list of lists which maps path ids from t1 to a list of path ids from t2
                   mergeable_paths[t1pi] = [t2pj, t2pk, t2pl, ...]
        '''
        t1_paths = self.all_paths[t1_id]
        t2_paths = self.all_paths[t2_id]

        t1_merges = []
        for p1 in t1_paths:  # for each path in tree 1
            p1_merges = []

            for i in range(len(t2_paths)):  # for each path in tree 2
                p2 = t2_paths[i]
                if self.is_mergable(p1, p2):
                    p1_merges.append(i)
            t1_merges.append(p1_merges)

        return t1_merges

    def is_mergable(self, p1, p2):
        p1_pred = p1[-1:, -1:][0][0]
        p2_pred = p2[-1:, -1:][0][0]

        # if both paths lead to leafs of the same class
        if p1_pred != p2_pred:
            return False
        else:
            p1_features = p1[:-1, 1:2]
            p2_features = p2[:-1, 1:2]
            shared_features = np.intersect1d(p1_features, p2_features)

            mergable = True
            # check that all shared features are resolveable collisions
            for feature_i in shared_features:
                mergable = mergable and self.is_resolveable(p1, p2, feature_i)
            return mergable

    def build_graphs(self, trees, nclasses):
        self.adjacencys = self.build_adjacencys(trees, nclasses)
        graphs = []
        for a in self.adjacencys:
            g = nx.Graph(a)
            graphs.append(g)
        self.graphs = graphs

    def build_adjacencys(self, trees, nclasses):
        ntrees = len(trees)
        adjacencys = []

        # create an adjacency matrix for each class, each matrix is the size of npaths x npaths (classwise)
        # an entry adjancy[i][j] indicates that those two paths are sythesizeable and should be connected in the graph
        for class_id in range(nclasses):
            adjacencys.append(
                np.zeros(shape=(self.npaths[class_id], self.npaths[class_id]), dtype=int))

        for t1_id in range(ntrees):  # for each tree
            for t2_id in range(ntrees):  # check every other tree pairwise
                # for every path in t1, find the set of paths that are sythesizeable with it in t2
                t1_t2_merges = self.sythesizable_paths[t1_id][t2_id]
                # for each path in t1 with at least one sythesizeable path in t2
                for p1_id in range(len(t1_t2_merges)):
                    t1p1_index = self.treepath_to_idx[t1_id][p1_id][1]  # index is classid, pathid
                    t1p1_class = int(self.all_paths[t1_id][p1_id][-1, 3])
                    # iterate over each sythesizeable path and connect them
                    for p2_id in t1_t2_merges[p1_id]:
                        t2p2_index = self.treepath_to_idx[t2_id][p2_id][1]
                        adjacencys[t1p1_class][t1p1_index][t2p2_index] = 1

        return adjacencys

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

    def is_resolveable(self, p1, p2, feature_i):
        '''
        Find the nodes in p1 and p2 which condition feature_i and check that they don't have conflicting conditions
        For two nodes n1 and n2 which condition feature i n1: x[i] <> a, n2: x[i] <> b, assuming a < b. Return false if the unresolveable condition n1: x[i] < a, n2: x[i] > b is found and true otherwise.
        '''
        idx1 = (p1[:, 1:2] == feature_i).squeeze()
        idx2 = (p2[:, 1:2] == feature_i).squeeze()

        all_resolveable = True

        for cond1, thresh1 in p1[idx1, 2:4]:
            for cond2, thresh2 in p2[idx2, 2:4]:
                if thresh1 < thresh2:
                    fails = (cond2 == 1) and (cond1 == 0)
                elif thresh1 == thresh2:
                    fails = (cond1 != cond2)
                elif thresh1 > thresh2:
                    fails = (cond1 == 1) and (cond2 == 0)

                if fails:
                    all_resolveable = False

        return all_resolveable

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

        # number of hyper-rectangles to index
        if hyperparameters.get("facet_nrects") is None:
            self.n_rects = 1000
            print("No facet_nrects provided, using {}".format(self.n_rects))
        else:
            self.n_rects = hyperparameters.get("facet_nrects")

        if hyperparameters.get("facet_sample") is None:
            self.sample_type = "Training"
            print("No facet_sample type provided, using training data")
        else:
            self.sample_type = hyperparameters.get("facet_sample")

        if hyperparameters.get("facet_enumerate") is None:
            self.enumeration_type = "PointBased"
            print("No facet_enumerate type provided, using PointBased")
        else:
            self.enumeration_type = hyperparameters.get("facet_enumerate")
            if self.enumeration_type not in ["PointBased", "GraphBased"]:
                print("{} not a valid facet_enumerate, defaulting to PointBased")
                self.enumeration_type = "PointBased"

        # print messages
        if hyperparameters.get("verbose") is None:
            self.verbose = False
        else:
            self.verbose = hyperparameters.get("verbose")