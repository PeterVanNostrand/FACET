# handle circular imports that result from typehinting
from __future__ import annotations

import bisect
# core python packages
import math
from typing import TYPE_CHECKING, Dict, List, Tuple
from detectors.gradient_boosting_classifier import GradientBoostingClassifier
from utilities.math_tools import sigmoid

# graph packages
import networkx as nx
# scientific and utility packages
import numpy as np
from sklearn import tree
from tqdm.auto import tqdm
from dataset import DataInfo

from detectors.random_forest import RandomForest
# from explainers.branching import BranchIndex
from explainers.explainer import Explainer
from explainers.bit_vector import BitVectorIndex
from explainers.bit_vector import LOWER, UPPER
# custom classes
from utilities.metrics import dist_euclidean
from baselines.ocean.CounterFactualParameters import FeatureType

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from manager import MethodManager


class FACETIndex(Explainer):
    # TODO Implement per axis minimum robustness paramater, and minimum robustness along all axes
    def __init__(self, manager, hyperparameters: dict):
        self.manager: MethodManager = manager
        self.model_type = manager.model_type
        self.parse_hyperparameters(hyperparameters)

    def save_tree_fig(self, t_id: int) -> None:
        plt.figure(dpi=300)
        t = self.manager.model.model.estimators_[0][t_id]
        tree.plot_tree(t)
        plt.savefig("tree_{:02d}.png".format(t_id))
        # plt.show()

    def prepare(self, xtrain=None, ytrain=None):
        data = xtrain
        if self.model_type == "RandomForest":
            model: RandomForest = self.manager.model
            trees = model.model.estimators_
            self.ntrees = len(trees)
            self.nclasses = model.model.n_classes_
            self.nfeatures = len(model.model.feature_importances_)
            self.rf_hardvoting = self.manager.model.hard_voting
            # the number of trees which much aggree for a prediction
            self.majority_size = math.floor(self.ntrees / 2) + 1
        elif self.model_type == "GradientBoostingClassifier":
            model: GradientBoostingClassifier = self.manager.model
            trees = [_[0] for _ in model.model.estimators_]
            self.ntrees = len(trees)
            self.nclasses = model.model.n_classes_
            self.nfeatures = len(model.model.feature_importances_)

        # walk each path and store their representation in an array
        self.all_paths, self.leaf_vals = self.build_paths(trees)
        # convert each leaf node into its corresponding hyper-rectangle
        self.leaf_rects = self.leaves_to_rects(self.all_paths, self.leaf_vals)

        if self.model_type == "GradientBoostingClassifier":
            self.leaf_extremes = self.find_leaf_extremes()

        # build the index of majority size hyper-rectangles
        if self.enumeration_type == "PointBased":
            self.initialize_index()
            self.point_enumerate(data)

        if self.search_type == "BitVector":
            self.build_bitvectorindex()

    def build_bitvectorindex(self):
        # create redundant bit vector index
        self.rbvs: List[BitVectorIndex] = []
        for class_id in range(self.nclasses):
            if self.verbose:
                print("class {}".format(class_id))
            self.rbvs.append(BitVectorIndex(rects=self.index[class_id],
                                            explainer=self, hyperparameters=self.hyperparameters))

    def prepare_dataset(self, x: np.ndarray, y: np.ndarray, ds_info: DataInfo) -> None:
        # create a copy of the DataInfo object
        self.ds_info: DataInfo = ds_info.copy()
        # make the possible values into numpy arrays
        for i in range(self.ds_info.ncols):
            if self.ds_info.possible_vals[i] != []:
                self.ds_info.possible_vals[i] = np.array(self.ds_info.possible_vals[i])
        # compute smart weights if needed for handling unscaled data
        if self.use_smart_weight:
            self.equal_weights = self.get_equal_weights()
        # if we're using smart weights and they're different from unweighted
        float_point_val = 1e-8
        if self.model_type == "GradientBoostingClassifier" and self.manager.model.maxdepth is None:
            float_point_val = 2e-8  # deep gradient boosting leads to tight ranges, use extra space
        if self.equal_weights is not None:
            self.EPSILONS = float_point_val * self.equal_weights
            self.offsets = self.offset_scalar * self.equal_weights
        # we're not using smart weighting, or the smart weights are equivalent to unweighted
        else:
            self.EPSILONS = np.tile(float_point_val, self.ds_info.ncols)
            self.offsets = np.tile(self.offset_scalar, self.ds_info.ncols)

    def compute_supports(self, data: np.ndarray):
        '''
        Computes the support for each vertex and each pairs of vertices in the classification output of the training data. That is fraction of all samples which are classificed into vertex Vi or pair of vertices Vi, Vj. These values are stored in self.vertex_support[i] and self.pairwise_support[i][j] respectively
        '''
        rf_detector: RandomForest = self.manager.model
        all_leaves = rf_detector.apply(data)  # leaves that each sample ends up in (nsamples in xtrain, ntrees)

        # compute the support of each vertex in the predictions of the training data
        # i.e. what fraction of the training predictions classify into each leaf
        vertex_supports = []
        pairwise_supports = []
        for class_id in range(self.nclasses):
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
                for vertex_j in range(vertex_i + 1, nvertices):
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

    def point_enumerate(self, data: np.ndarray) -> None:
        '''
        Uses the provided set of data to select a set of points in the input space, then selects and indexes one hyper-rectangle around each point
        '''
        tries = 0
        while len(self.index[0]) == 0 or len(self.index[1]) == 0:
            tries += 1
            if tries > 1:
                print("bad luck finding rects, trying again...")
            rect_points = self.select_points(training_data=data)
            self.index_rectangles(data=rect_points)

        # ! DEBUG START
        # if not self.check_rects_one_hot_valid():
        #     print("WARNING INVALID HYPERRECTS IN INDEX")
        # ! DEBUG END

    def one_hot_valid(self, rect: np.ndarray) -> bool:
        for cat_column_name, sub_col_idxs in self.ds_info.one_hot_schema.items():
            # requires more than one one-hot high
            n_set_high = sum(rect[sub_col_idxs, LOWER] > 0)
            if n_set_high > 1:
                return False
            # requires all one-hot low
            n_set_low = sum(rect[sub_col_idxs, UPPER] < 1)
            if n_set_low == len(sub_col_idxs):
                return False
        return True

    def check_rects_one_hot_valid(self) -> bool:
        invalid_count = 0
        for class_id in [0, 1]:
            for rect in self.index[class_id]:
                for cat_column_name, sub_col_idxs in self.ds_info.one_hot_schema.items():
                    n_set_high = sum(rect[sub_col_idxs, LOWER] > 0)
                    n_set_low = sum(rect[sub_col_idxs, UPPER] < 1)
                    if n_set_high > 1:
                        invalid_count += 1
                    elif (n_set_low == len(sub_col_idxs)):
                        invalid_count += 1
        return invalid_count == 0

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
                print("all same")
                # create a set of points randomly placed with a uniform distribution along each axis
                rect_points = np.random.uniform(low=0.0, high=1.0, size=(self.n_rects, training_data.shape[1]))
                # check to make sure the resulting set has points of both classes, redraw if needed
                preds = self.manager.predict(rect_points)
                all_same_class = len(np.unique(preds)) < 2
        elif self.sample_type == "Augment":
            all_same_class = True
            while all_same_class:
                # take a bootstrap sample of the training data
                rand_idxs = np.random.randint(low=0, high=training_data.shape[0], size=self.n_rects)
                rect_points = training_data[rand_idxs]
                # augment these points with random normally distributed noise
                noise = np.random.normal(loc=0.0, scale=self.standard_dev, size=rect_points.shape)
                rect_points += noise
                # check to make sure the resulting set has points of both classes, redraw if needed
                preds = self.manager.predict(rect_points)
                all_same_class = len(np.unique(preds)) < 2
        return rect_points

    def explore_index(self, nfeatures: int = 0, npoints: int = 1000, points: np.ndarray = None) -> float:
        '''
        Randomly samples a number of points and checks what percentage of those points fall in the index
        '''
        # generate points if not provided
        if points is None:
            if nfeatures == 0:
                nfeatures = self.nfeatures
            points = np.random.uniform(low=0.0, high=1.0, size=(npoints, nfeatures))
        # check the points
        num_inside = 0
        for p in points:
            if self.inside_index(p):
                num_inside += 1
        percent_covered = num_inside / points.shape[0]
        return percent_covered

    def mean_rect_sizes(self):
        mean_size = np.zeros(shape=(self.nclasses, self.nfeatures))
        # mean_noninf_size = np.zeros(shape=(self.nclasses, self.rf_nfeatures))
        for label in range(self.nclasses):
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
        test_rect[:, LOWER][test_rect[:, LOWER] == -np.inf] = 0
        test_rect[:, UPPER][test_rect[:, UPPER] == np.inf] = 1
        widths = test_rect[:, UPPER] - test_rect[:, LOWER]
        return widths

    def inside_index(self, point: np.ndarray) -> bool:
        '''
        Checks if the given point falls within a hyper-rectangle included in the index
        '''
        label = self.manager.predict([point])
        covered = False
        i = 0
        while i < len(self.index[label]) and not covered:
            covered = covered or self.is_inside(point, self.index[label][i])
            i += 1
        return covered

    def leaves_to_rects(self, all_paths: List[List[np.ndarray]], leaf_vals: List[List[np.ndarray]]) -> List[Dict]:
        '''
        For each path in the ensemble, identifies the leaf node of that path and builds the corresponding hyper-rectangle

        Parameters
        ----------
        The enumerated paths from build_paths, a ragged list of (ntrees, npaths_i)

        Returns
        -------
        leaf_rects: a list of dictionarys where leaft_rects[i][j] returns the (leaf_class, rect) corresponding to the path ending in scikit node id j from tree i
        '''
        leaf_rects = [{} for _ in range(self.ntrees)]
        for tid in range(self.ntrees):
            for path, val in zip(all_paths[tid], leaf_vals[tid]):
                # get the scikit node id of the leaf and its class
                leaf_node_id = int(path[-1, 0])
                # build and save the hyper-rectangle that corresponds to this leaf
                rect = self.leaf_rect(path)
                if self.model_type == "RandomForest":
                    leaf_class = int(path[-1, -1])
                    # save the class and the hyper-rectangle
                    leaf_rects[tid][leaf_node_id] = (leaf_class, rect, val)
                elif self.model_type == "GradientBoostingClassifier":
                    leaf_val = path[-1, -1]
                    # save the class and the hyper-rectangle
                    leaf_rects[tid][leaf_node_id] = (leaf_val, rect, val)
        return leaf_rects

    def index_rectangles(self, data: np.ndarray):
        '''
        This method uses the training data to enumerate a set of hyper-rectangles of each classes and adds them to an index for later searching during explanation
        '''
        preds = self.manager.predict(data)
        model = self.manager.model
        # get the leaves that each sample ends up in
        all_leaves = model.apply(data).reshape(data.shape[0], self.ntrees)  # shape (nsamples in xtrain, ntrees)

        visited_rects = [{} for _ in range(self.nclasses)]  # a hashmap to store which rectangles we have visited
        # for each instance in the training set
        for instance, label, leaf_ids in zip(data, preds, all_leaves):
            rect, paths_used = self.enumerate_rectangle(leaf_ids, label)
            key = hash(tuple(paths_used))  # hash the path list to get a unique key corresponding to this HR
            if key not in visited_rects[label]:  # if we haven't visited this hyper-rectangle before
                if self.one_hot_valid(rect):
                    self.add_to_index(label, rect)  # add it to the index
                    visited_rects[label][key] = True  # remember that we've indexed it

    def add_to_index(self, label: int, rectangle: np.ndarray) -> None:
        '''
        Add the rectangle to the index
        '''
        self.index[label].append(rectangle)

    def initialize_index(self) -> None:
        '''
        Creates an empty index to store the hyper-rectangles. Functionized to allow for multiple indexing options
        '''
        self.index = [[] for _ in range(self.nclasses)]

    def enumerate_rectangle(self, leaf_ids: List[int], label: int) -> np.ndarray:
        '''
        Given a set of scikitlearns leaf_ids, one per tree in the ensemble, extracts the paths which correspond to these leaves and constructs a majority size hyper-rectangle from their intersection

        Paramters
        ---------
        leaf_ids: a list of integer values of size ntrees, where leaf_ids[i] is the scikit learn integer id of the leaf node selected by tree i

        Returns
        -------
        rect: a numpy ndarray of shape (ndim, 2) containing a majority size hyper-rectangle
        paths_used: a list of pairs (tree_id, leaf_id) which correspond to the leaves the form the intersection
        '''
        all_bounds: List[np.ndarray] = []  # list of leaf hyper-rectangles w/ dims (nfeatures, 2)
        paths: List[(int, int)] = []  # list of tree_id, leaf_id included in all_bounds
        path_probs: List[np.ndarray] = []  # list of class probs for the given leaves, dims (nclasses,)
        for tree_id in range(self.ntrees):
            leaf_class, leaf_rect, class_probs = self.leaf_rects[tree_id][leaf_ids[tree_id]]
            if self.model_type == "RandomForest":
                # for hard voting take only the hyper-rectangles which predict the given label
                # for soft voting the probability of all classes for all leaves contributes to the final classfication
                # e.g. several leaves predict class 1 with probs [0.49, 0.51], we may need them to make class 0 pred
                use_path = not self.rf_hardvoting or (self.rf_hardvoting and leaf_class == label)
            elif self.model_type == "GradientBoostingClassifier":
                # in GBC we need to consider all paths as path's don't have classes in the same way
                use_path = True

            if use_path:
                paths.append((tree_id, leaf_ids[tree_id]))
                all_bounds.append(leaf_rect)
                path_probs.append(class_probs)
        path_probs = np.vstack(path_probs)
        rect, paths_used = self.select_intersection(all_bounds, paths, path_probs, label)
        return rect, paths_used

    def select_hard_intersection(self, all_bounds: List[np.ndarray], paths: List[Tuple[int]], path_probs: np.ndarray, label: int) -> np.ndarray:
        '''
        ensemble is using majority vote take the intersection of the first nmajority hyper-rectangles, we should only receive leaf hyper-rectangles of class `label`
        '''
        # initialize the hyper-rectangle as unbounded on both side of all axes
        rect = np.zeros((self.nfeatures, 2))
        rect[:, LOWER] = -np.inf
        rect[:, UPPER] = np.inf

        if self.intersect_order == "Axes":
            # compute how many axes each rectangle bounds, this is equal to the number of noninfinite threshold
            # values in the hyper-rectangle array. Ranges [0, 2*ndims]
            n_bounded_axes = [0] * len(all_bounds)
            for i in range(len(all_bounds)):
                n_bounded_axes[i] = np.isfinite(all_bounds[i]).sum()
            order = np.argsort(n_bounded_axes)  # take rects with the fewest bounds first
        elif self.intersect_order == "Size":
            mean_rect_widths = [0] * len(all_bounds)
            for i in range(len(all_bounds)):
                mean_rect_widths[i] = self.rect_width(all_bounds[i]).mean()
            order = np.argsort(mean_rect_widths)  # take largest rects first
        elif self.intersect_order == "Ensemble":
            order = list(range(len(all_bounds)))  # take rects in order of trees in ensemble
        elif self.intersect_order == "Probability":  # take rects in order of largest probability
            order = path_probs[:, label].argsort()[::-1]

        i = 0
        paths_used = []
        while i < len(all_bounds) and i < self.majority_size:
            rect[:, LOWER] = np.maximum(rect[:, LOWER], all_bounds[order[i]][:, LOWER])  # intersection of minimums
            rect[:, UPPER] = np.minimum(rect[:, UPPER], all_bounds[order[i]][:, UPPER])  # intersection of maximums
            bisect.insort(paths_used, paths[order[i]])  # remember which leaves we've used, keep sorted asc by tid
            i += 1

        return rect, paths_used

    def gbc_intersect_all(self, leaf_rects: List[np.ndarray], leaf_vals: np.ndarray) -> Tuple[np.ndarray, float]:
        '''
        returns the intersection of all the leaf rectangles and the accumulated odds of that intersection
        '''
        # initialize the hyper-rectangle as unbounded on both side of all axes
        rect = np.zeros((self.nfeatures, 2))
        rect[:, LOWER] = -np.inf
        rect[:, UPPER] = np.inf
        # take the interesection
        accumulated_odds = self.manager.model.init_value
        for i in range(len(leaf_rects)):
            rect[:, LOWER] = np.maximum(rect[:, LOWER], leaf_rects[i][:, LOWER])  # itersect mins
            rect[:, UPPER] = np.minimum(rect[:, UPPER], leaf_rects[i][:, UPPER])  # itersect maxs
            accumulated_odds += self.manager.model.lr * leaf_vals[i, 0]
        return rect, accumulated_odds

    def find_leaf_extremes(self) -> List[Tuple[float, float]]:
        worst_values = []
        for tree_id in range(self.ntrees):
            lowest_val = np.inf
            highest_val = -np.inf
            for path_id in range(len(self.all_paths[tree_id])):
                val = self.all_paths[tree_id][path_id][-1, -1]
                if val < lowest_val:
                    lowest_val = val
                if val > highest_val:
                    highest_val = val
            worst_values.append([lowest_val, highest_val])
        return worst_values

    def gbc_accumulate_odds(self, leaf_vals: np.ndarray) -> float:
        return self.manager.model.init_value + self.manager.model.lr * sum(leaf_vals)[0]

    def select_gbc_intersection_minimal(self, leaf_rects: List[np.ndarray], paths: List[Tuple[int]], leaf_vals: np.ndarray, label: int) -> np.ndarray:
        '''
        Given a set of of leaf rectangles, their corresponding paths, and their leaf values. Examine the set of leaves and intersect as few as needed to ensure that the intersection is guaranteed to be a counterfactual region of the observed class
        '''

        # if we intersect all the leaf rects, we're guranteed to be class homogenous
        odds = self.gbc_accumulate_odds(leaf_vals)
        # odds > 0 correspond to class one, odds < 0 correspond to class zero
        is_class_one = (odds > 0)  # raw odds, run through sigmoid(odds) to get class probs
        bad_direction = 0 if is_class_one else 1  # for [lowest_val, highest_val] of leaf_extremes
        # we then start from the end and drop leaf rects and assume that its replaced with the worst
        next_tree = self.ntrees - 1
        bad_vals = []
        while ((is_class_one and odds > 0) or (not is_class_one and odds < 0)) and next_tree > 0:
            # drop the last leaf
            dropped_odds = self.gbc_accumulate_odds(leaf_vals[0:next_tree])
            # replace it with the worse possible leaf from that tree
            bad_vals.append(self.leaf_extremes[next_tree][bad_direction])
            # compute the worst case odds
            odds = dropped_odds + self.manager.model.lr * sum(bad_vals)
            # move to the next tree
            next_tree -= 1
        if next_tree != (self.ntrees - 1):
            next_tree += 1
        rect, accumulated_odds = self.gbc_intersect_all(leaf_rects[0:next_tree+1], leaf_vals[0:next_tree+1])
        used_paths = paths[0:next_tree+1]
        # !DEBUG - CHECK THAT THE LEAF CLASS PROBS MATCH THE PREDICATED PRBS
        rect_instance = self.fit_to_rectangle(np.zeros(shape=(self.nfeatures)), rect)
        if rect_instance is not None:
            pred = self.manager.model.model.predict([rect_instance])[0]
            if pred != label:
                print("ERROR CREATING RECT")
        # !DEBUG END
        return rect, used_paths

    def select_gbc_intersection_complete(self, leaf_rects: List[np.ndarray], paths: List[Tuple[int]], leaf_vals: np.ndarray, label: int) -> np.ndarray:
        '''
        Given a set of of leaf rectangles, their corresponding paths, and their leaf values. Intersect all the leaf rectangles to create a counterfactual region of the observed class
        '''
        rect, accumulated_odds = self.gbc_intersect_all(leaf_rects=leaf_rects, leaf_vals=leaf_vals)
        # !DEBUG - CHECK THAT THE LEAF CLASS PROBS MATCH THE PREDICATED PRBS
        # class_one_prob = sigmoid(accumulated_odds)
        # class_probs = [1.0 - class_one_prob, class_one_prob]
        leaf_instance = self.fit_to_rectangle(np.zeros(shape=(self.nfeatures)), rect)
        if leaf_instance is not None:
            pred = self.manager.model.model.predict([leaf_instance])[0]
            if pred != label:
                print("ERROR CREATING RECT")
        # !DEBUG END
        return rect, paths

    def select_soft_intersection(self, all_bounds: List[np.ndarray], paths: List[Tuple[int]], path_probs: np.ndarray, label: int) -> np.ndarray:
        '''
        ensemble is using soft voting, take intersection of sufficient hyper-rects to reach a majority probability, we sould receive ntrees leaf hyper-rectangles
        '''

        # initialize the hyper-rectangle as unbounded on both side of all axes
        rect = np.zeros((self.nfeatures, 2))
        rect[:, LOWER] = -np.inf
        rect[:, UPPER] = np.inf
        paths_used = []
        accumulated_prob = np.zeros(shape=(self.nclasses,))

        if self.intersect_order == "Probability":
            # take rects in order of largest probability
            order = path_probs[:, label].argsort()[::-1]
            i = 0
            while i < len(all_bounds) and accumulated_prob[label] <= 0.5:
                rect[:, LOWER] = np.maximum(rect[:, LOWER], all_bounds[order[i]][:, LOWER])  # intersection of minimums
                rect[:, UPPER] = np.minimum(rect[:, UPPER], all_bounds[order[i]][:, UPPER])  # intersection of maximums
                bisect.insort(paths_used, paths[order[i]])  # remember which leaves we've used, keep sorted asc by tid
                accumulated_prob += (1 / self.ntrees) * path_probs[order[i]]
                i += 1

        else:  # we use an order other than largest probability
            # separate the leaf hyper-rectangles by whether they match the desired class
            idx_match_label = path_probs[:, label] > 0.5
            all_bounds = np.vstack(all_bounds).reshape(self.ntrees, self.nfeatures, 2)
            paths = np.vstack(paths)

            match_bounds = all_bounds[idx_match_label]
            match_paths = paths[idx_match_label]
            match_probs = path_probs[idx_match_label]

            other_bounds = all_bounds[~idx_match_label]
            other_paths = paths[~idx_match_label]
            other_probs = path_probs[~idx_match_label]

            if self.intersect_order == "Axes":
                n_bounded_axes = [0] * len(match_bounds)
                for i in range(len(match_bounds)):
                    n_bounded_axes[i] = np.isfinite(match_bounds[i]).sum()
                match_order = np.argsort(n_bounded_axes)  # take rects with the fewest bounds first
            elif self.intersect_order == "Size":
                mean_rect_widths = [0] * len(match_bounds)
                for i in range(len(match_bounds)):
                    mean_rect_widths[i] = self.rect_width(match_bounds[i]).mean()
                match_order = np.argsort(mean_rect_widths)  # take largest rects first
            elif self.intersect_order == "Ensemble":
                match_order = list(range(len(match_bounds)))  # take rects in order of trees in ensemble

            # start with intersecting the hyper-rectangles which have the matching class
            i = 0
            while i < len(match_bounds) and accumulated_prob[label] <= 0.5:
                rect[:, LOWER] = np.maximum(rect[:, LOWER], match_bounds[match_order[i]][:, LOWER])  # itersect mins
                rect[:, UPPER] = np.minimum(rect[:, UPPER], match_bounds[match_order[i]][:, UPPER])  # itersect maxs
                bisect.insort(paths_used, tuple(match_paths[match_order[i]]))  # remember leaves we used
                accumulated_prob += (1 / self.ntrees) * match_probs[match_order[i]]
                i += 1

            # on rare occassion this intersection will not have sufficient probability for the desired class
            # if needed continue intersecting leaves of the non-desired class until the desired class is the majority
            nm_order = other_probs[:, label].argsort()[::-1]  # take other leaf with highest desired class prob first
            i = 0
            while i < len(other_bounds) and accumulated_prob[label] <= 0.5:
                rect[:, LOWER] = np.maximum(rect[:, LOWER], other_bounds[nm_order[i]][:, LOWER])  # itersect mins
                rect[:, UPPER] = np.minimum(rect[:, UPPER], other_bounds[nm_order[i]][:, UPPER])  # itersect maxs
                bisect.insort(paths_used, tuple(other_paths[nm_order[i]]))  # remember leaves we used
                accumulated_prob += (1 / self.ntrees) * other_probs[nm_order[i]]
                i += 1

        return rect, paths_used

    def select_intersection(self, all_bounds: List[np.ndarray], paths: List[(int, int)], path_probs: np.ndarray, label: int) -> np.ndarray:
        '''
        Given a list of ntrees hyper-rectangles each corresponing to a leaf node, constructs a majority size hyper-rectangle by taking the intersection of nmajority leaf node hyper-rectangles

        Parameters
        ----------
        all_bounds: a list of ntrees numpy arrays, each of size (nfeatures, 2) generated by leaf_rect
        paths: list[(int, int)] a list of pairs (tree_id, leaf_id) corresponding to the leaves included in all_bounds

        Returns
        -------
        rect: a numpy ndarray of shape (ndim, 2) containing a majority size hyper-rectangle
        paths_used: a list of pairs (tree_id, leaf_id) which correspond to the leaves the form the intersection
        '''
        if self.model_type == "RandomForest":
            if self.rf_hardvoting:
                return self.select_hard_intersection(all_bounds, paths, path_probs, label)
            else:
                return self.select_soft_intersection(all_bounds, paths, path_probs, label)
        elif self.model_type == "GradientBoostingClassifier":
            if self.gbc_intersect_order == "CompleteEnsemble":
                return self.select_gbc_intersection_complete(all_bounds, paths, path_probs, label)
            elif self.gbc_intersect_order == "MinimalWorstGuess":
                return self.select_gbc_intersection_minimal(all_bounds, paths, path_probs, label)

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
        feature_bounds = np.zeros((self.nfeatures, 2))
        feature_bounds[:, LOWER] = -np.inf
        feature_bounds[:, UPPER] = np.inf

        for i in range(path[:-1, :].shape[0]):
            feature = int(path[i, 1])
            condition = int(path[i, 2])
            threshold = path[i, 3]
            if condition == 0:  # less than or equal to, update max value to be lower
                feature_bounds[feature][1] = min(threshold, feature_bounds[feature][1])
            else:  # greater than, update min value to be higher
                feature_bounds[feature][0] = max(threshold, feature_bounds[feature][0])
        return feature_bounds

    def build_paths(self, trees: List[tree.DecisionTreeClassifier]) -> List[List[np.ndarray]]:
        '''
        Walks each tree and extracts each path from root to leaf into a data structure. Each tree is represented as a list of paths, with each path stored into an array

        Returns
        -------
        all_paths: a list of length ntrees, where all_paths[i] contains a list of the paths in tree i each represented by a numpy array as generated by in_order_path
        path_class_vals: the value of the leaf for each path. for random forest this is a list of class probabilities (nclasses,), for gradient boosting classifier its the logodds of the leaf
        '''
        ntrees = len(trees)
        all_paths = [[] for _ in range(ntrees)]
        path_class_vals = [[] for _ in range(ntrees)]
        for i in range(ntrees):
            all_paths[i], path_class_vals[i] = self.__in_order_path(t=trees[i], built_paths=[], leaf_vals=[])

        return all_paths, path_class_vals

    def __in_order_path(self, t, built_paths: List[np.ndarray] = [], leaf_vals: List[np.ndarray] = [], node_id=0, path: List = []):
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
            self.__in_order_path(t, built_paths, leaf_vals, node_id=t.tree_.children_left[node_id], path=left_path)

            # process right node, conditioned (>)
            right_path = path.copy()
            right_path.append([node_id, feature, 1, threshold])
            self.__in_order_path(t, built_paths, leaf_vals, node_id=t.tree_.children_right[node_id], path=right_path)

            return built_paths, leaf_vals

        else:  # this is a leaf node
            leaf_val: np.ndarray = t.tree_.value[node_id]
            path = path.copy()
            # get the class probabilities for this leaf
            if self.model_type == "RandomForest":
                samps_per_class = leaf_val
                class_id = np.argmax(samps_per_class)
                path.append([node_id, -1, -1, class_id])
                class_probs: np.ndarray = (samps_per_class / samps_per_class.sum()).squeeze()
                leaf_vals.append(class_probs)
            elif self.model_type == "GradientBoostingClassifier":
                leaf_odds = leaf_val[0, 0]
                path.append([node_id, -1, -1, leaf_odds])
                leaf_vals.append(leaf_odds)

            # store the completed path and exit
            finished_path = np.array(path)
            built_paths.append(finished_path)

            return built_paths, leaf_vals

    def is_inside(self, x: np.ndarray, rect: np.ndarray) -> bool:
        '''
        Returns true if x falls inside the given rectangle, false otherwise
        '''
        return (x >= rect[:, LOWER]).all() and (x <= rect[:, UPPER]).all()

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
        # use epsilon to account for floating point imprecision for rare values

        # if we're using unscaled values, we may have very large values with large float. pt. error

        xprime = x.copy()
        binary_required_high = set()
        binary_required_low = set()
        if self.ds_info.all_numeric:  # if all features are numeric, do a quick fit using array operations
            # determine which values need to adjusted to be smaller, and which need to be larger
            low_values = xprime <= (rect[:, LOWER] + self.EPSILONS)
            high_values = xprime >= (rect[:, UPPER] - self.EPSILONS)

            # check that the offset will not overstep the min or max value, this occurs when the range between
            # a min and max value is less than the offset size
            rect_width = (rect[:, UPPER] - rect[:, LOWER])
            overstep_range = rect_width <= self.offsets
            # identify features which need to be adjusted and will overstep the min or max value
            idx_overstep = np.logical_and(overstep_range, np.logical_or(low_values, high_values))

            # increase low feature values to one offset above min
            xprime[low_values] = rect[low_values, LOWER] + self.offsets[low_values]
            # decrease high features one to one offset below min
            xprime[high_values] = rect[high_values, UPPER] - self.offsets[high_values]
            # for oversteped bounds use the average between min and max
            xprime[idx_overstep] = rect[idx_overstep, LOWER] + rect_width[idx_overstep] / 2
        else:  # if there are non-numeric features, handle them directly
            i = 0
            while i < self.ds_info.ncols:
                # determine if the value is too low, too high, and if an overstep will occur
                is_low = (xprime[i] <= (rect[i, LOWER] + self.EPSILONS[i]))
                is_high = (xprime[i] >= (rect[i, UPPER] - self.EPSILONS[i]))
                rect_width_i = (rect[i, UPPER] - rect[i, LOWER])
                is_overstep = (rect_width_i <= self.offsets[i])

                # for numeric features do a simple step if needed with overstep checking
                if self.ds_info.col_types[i] == FeatureType.Numeric:
                    if is_overstep:
                        xprime[i] = rect[i, LOWER] + rect_width_i / 2
                    elif is_low:
                        xprime[i] = rect[i, LOWER] + self.offsets[i]
                    elif is_high:
                        xprime[i] = rect[i, UPPER] - self.offsets[i]
                # for binary features flip the value if needed
                elif self.ds_info.col_types[i] == FeatureType.Binary:
                    if rect[i, UPPER] < 1.0:
                        binary_required_low.add(i)
                    if rect[i, LOWER] > 0.0:
                        binary_required_high.add(i)
                    if is_low:
                        # since some binary columns are from one-hot encoding of categorical features
                        # we need to enfore the one-hot property for these features
                        if i in self.ds_info.reverse_one_hot_schema:  # if this column is for a one-hot encoding
                            # determine which feature it encodes
                            feat_name = self.ds_info.reverse_one_hot_schema[i]
                            # clear the other columns for this feature
                            xprime[self.ds_info.one_hot_schema[feat_name]] = 0.0
                        # set the current column value if needed
                        xprime[i] = 1.0
                    elif is_high:
                        # set the feature low
                        xprime[i] = 0.0
                        binary_required_low.add(i)
                        if i in self.ds_info.reverse_one_hot_schema:  # if this column is for a one-hot encoding
                            # determine which feature it encodes
                            feat_name = self.ds_info.reverse_one_hot_schema[i]
                            # check which other columns are set
                            feature_columns = np.array(self.ds_info.one_hot_schema[feat_name])
                            n_set_cols = sum(xprime[feature_columns])
                            if n_set_cols == 0:  # if this was the only col set, we need to set one
                                rect_allowed_high = feature_columns[rect[feature_columns, UPPER] >= 1.0]
                                xprime[np.random.choice(rect_allowed_high, 1)] = 1.0

                # for discrete values find the next largest or next smallest value as needed
                elif self.ds_info.col_types[i] == FeatureType.Discrete:
                    if is_low:  # if low, choose the next largest value
                        differences = (self.ds_info.possible_vals[i] - rect[i, LOWER])
                        differences[differences < 0] = np.inf
                        idx_next_larger = np.argmin(differences)
                        # if we're in danger of floating point innacuracy
                        if (self.ds_info.possible_vals[i][idx_next_larger] - rect[i, LOWER]) < self.EPSILONS[i]:
                            if idx_next_larger < len(self.ds_info.possible_vals[i]) - 1:  # and we can step up, do it
                                idx_next_larger += 1
                            else:  # if we can't this is an invalid explanation
                                return None
                        xprime[i] = self.ds_info.possible_vals[i][idx_next_larger]
                        if xprime[i] > rect[i, UPPER] - self.EPSILONS[i]:  # if the next step up is out of the rect
                            return None
                    elif is_high:  # if high, choose the next smalles value
                        differences = (self.ds_info.possible_vals[i] - rect[i, UPPER])
                        differences[differences > 0] = -np.inf
                        idx_next_lower = np.argmax(differences)
                        # if we're in danger of floating point innacuracy
                        if (rect[i, UPPER] - self.ds_info.possible_vals[i][idx_next_lower]) < self.EPSILONS[i]:
                            if idx_next_lower > 1:  # and we can step down, do it
                                idx_next_lower -= 1
                            else:  # if we can't this is an invalid explanation
                                return None
                        xprime[i] = self.ds_info.possible_vals[i][idx_next_lower]
                        if xprime[i] < rect[i, LOWER] + self.EPSILONS[i]:  # if the next step down is out of the rect
                            return None
                # handle categorical one-hot encoded values, making sure only one column is hot for categorical feature
                elif self.ds_info.col_types[i] == FeatureType.Categorical:
                    print("How'd you get here? Categorical features should be one-hot encoded")
                    print("If for some reason you don't want to one-hot encode please consider marking as discrete instead")
                i += 1
        # !DEBUG START
        # if not self.ds_info.check_valid([xprime]):
        #     print("CRITICAL ERROR - FACET GENERATED AN INVALID EXPLANATION")
        # if not self.is_inside(xprime, rect):
        #     print("ERROR, COUNTERFACTUAL EXAMPLE NOT IN REGION")
        # ! DEBUG END

        return xprime

    def rect_center(self, rect: np.ndarray) -> np.ndarray:
        '''
        Returns the center point of the given rectangle, assuming bounds of +-inf are 1.0 and 0.0 respectively
        '''
        lower_bounds = np.maximum(rect[:, LOWER], -10.0)
        upper_bounds = np.minimum(rect[:, UPPER], 10.0)
        center = (upper_bounds + lower_bounds) / 2
        return center

    def get_equal_weights(self) -> np.ndarray:
        weights = None
        have_numeric = FeatureType.Numeric in self.ds_info.col_types
        have_discrete = FeatureType.Discrete in self.ds_info.col_types
        have_unscaled_num = (have_numeric and not self.ds_info.normalize_numeric)
        have_unscaled_disc = (have_discrete and not self.ds_info.normalize_discrete)

        # if there are unscaled features
        if have_unscaled_num or have_unscaled_disc:
            # default all feature to weight of one
            weights = np.tile(1.0, (self.ds_info.ncols,))
            for col_id in range(self.ds_info.ncols):
                col_type = self.ds_info.col_types[col_id]
                if col_type == FeatureType.Numeric and have_unscaled_num:  # if we col is unscaled numeric data
                    weights[col_id] += (self.ds_info.col_scales[col_id][1] - self.ds_info.col_scales[col_id][0])
                elif col_type == FeatureType.Discrete and have_unscaled_disc:  # if col is unscaled discrete
                    weights[col_id] += (self.ds_info.col_scales[col_id][1] - self.ds_info.col_scales[col_id][0])
        return weights

    def explain(self, x: np.ndarray, y: np.ndarray, k: int = 1, constraints: np.ndarray = None, weights: np.ndarray = None, max_dist: float = np.inf, min_robust: float = None, min_widths: np.ndarray = None, opt_robust: bool = False) -> np.ndarray:
        '''
        Parameters
        ----------
        `x`               : an array of samples each of dimensions (nsamples, nfeatures)
        `y`               : an array of predicted labels which correspond to the labels, (nsamples,)
        `k`               : the number of explanations requested
        `contraints`      : an array of shape (nfeatures, 2) where constraints[i][0/1] represents the
                            smallest/largest allowed value for feature i
        `weights`         : an array of shape (nfeatures,) corresonding to the ease of changing a feature
                            weights[i]=1 indicates normal cost and weights[i]>1 an easier cost
        `max_dist`        : the maximum distance from `x` to search for an explanation
        `min_robust`      : the minimum radial robustness an explanation must meet, applied to all features
        `min_widths`      : array of shape (features,) where min_widths[i] is the min required robustness of xprime[i]
        `opt_robust`      : When true chose a point in the nearest rect that maximizes robustness rather than min dist

        Returns
        -------
        `xprime` : a list of of counterfactual example arrays each with dimensions (nsamples, nfeatures)
        '''
        xprime = []  # an array for the constructed contrastive examples

        # assumimg binary classification [0, 1] set counterfactual class
        counterfactual_classes = ((y - 1) * -1)

        # check if we need to weight for unscaled features
        if weights is None:
            weights = self.equal_weights

        if self.search_type == "Linear":
            # performs, a linear scan of all the hyper-rectangles
            # does not support k, weights, constraints, max_dist, min_robust, min_widths
            for i in range(x.shape[0]):
                nearest_rect = None
                min_dist = np.inf
                # find the indexed rectangle of the the counterfactual class that is cloest
                for rect in self.index[counterfactual_classes[i]]:
                    test_instance = self.fit_to_rectangle(x[i], rect)
                    if test_instance is not None:
                        dist = self.distance_fn(x[i], test_instance)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_rect = rect
                # generate a counterfactual example which falls within this rectangle
                if opt_robust:
                    explanation = self.rect_center(nearest_rect)
                else:
                    explanation = self.fit_to_rectangle(x[i], nearest_rect)
                xprime.append(explanation)

        elif self.search_type == "BitVector":
            progress = tqdm(total=x.shape[0], desc="FACETIndex", leave=False)
            for i in range(x.shape[0]):  # for each instance
                nearest_rect = None
                result = self.rbvs[counterfactual_classes[i]].point_query(
                    instance=x[i],
                    constraints=constraints,
                    weights=weights,
                    k=k,
                    max_dist=max_dist,
                    min_robust=min_robust,
                    min_widths=min_widths
                )
                if k == 1 and result is not None:
                    nearest_rect = result
                    if opt_robust:
                        explanation = self.rect_center(nearest_rect)
                    else:
                        explanation = self.fit_to_rectangle(x[i], nearest_rect)
                    # !DEBUG START
                    # check_class = self.manager.predict([explanation])[0]
                    # if check_class != counterfactual_classes[i]:
                    #     print("failed explanation")
                    #     a = self.fit_to_rectangle(x[i], nearest_rect)
                    #     print("idx: {}, desired: {}, observed: {}".format(i, counterfactual_classes[i], check_class))
                    #     print(x[i])
                    # !DEBUG END
                elif k > 1 and len(result) > 0:
                    nearest_rect = result[0]
                    if opt_robust:
                        explanation = self.rect_center(nearest_rect)
                    else:
                        explanation = self.fit_to_rectangle(x[i], nearest_rect)
                else:
                    explanation = [np.inf for _ in range(x.shape[1])]

                xprime.append(explanation)
                progress.update()
            progress.close()

        # swap np.inf (no explanatio found) for zeros to allow for prediction on xprime
        xprime = np.array(xprime)
        idx_inf = (xprime == np.inf).any(axis=1)
        xprime[idx_inf] = np.tile(0, x.shape[1])
        # check that all counterfactuals result in a different class
        preds = self.manager.predict(xprime)
        failed_explanation = (preds == y)
        xprime[failed_explanation] = np.tile(np.inf, x.shape[1])
        # replace infinite values for invalid explanation
        xprime[idx_inf] = np.tile(np.inf, x.shape[1])

        if self.verbose:
            print("failed x':", failed_explanation.sum())

        return xprime

    def find_synthesizeable_paths(self, trees):
        ntrees = len(trees)
        sythesizable_paths = [[[] for _ in range(ntrees)] for _ in range(ntrees)]
        for i in range(ntrees):
            for k in range(ntrees):
                if (i != k):
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

    def parse_param(self, param: str, default_val):
        '''
        Check self.params for the given value and return it, if not present return the default value
        '''
        if param in self.params:
            return self.params[param]
        else:
            print("No {} set, using default {}={}".format(param, param, default_val))
            return default_val

    def parse_hyperparameters(self, hyperparameters: dict) -> None:
        self.hyperparameters = hyperparameters
        self.params: dict = hyperparameters.get("FACETIndex")

        # distance metric for explanation
        self.distance_fn = dist_euclidean

        # threshold offest for picking new values
        self.offset_scalar = self.parse_param("facet_offset", 0.001)
        self.n_rects = self.parse_param("facet_nrects", 20_000)

        self.offset_scalar = self.parse_param("facet_offset", 0.001)
        self.n_rects = self.parse_param("facet_nrects", 20_000)
        self.sample_type = self.parse_param("facet_sample", "Training")
        self.enumeration_type = self.parse_param("facet_enumerate", "PointBased")
        self.search_type = self.parse_param("facet_search", "Linear")
        self.verbose = self.parse_param("facet_verbose", False)
        self.standard_dev = self.parse_param("facet_sd", 0.1)
        self.intersect_order = self.parse_param("facet_intersect_order", "Probability")
        self.gbc_intersect_order = self.parse_param("gbc_intersection", "MinimalWorstGuess")

        if self.params.get("facet_smart_weight") is None:
            print("no facet_smart_weight, usinge True")
            self.use_smart_weight = True
        else:
            self.use_smart_weight = self.params.get("facet_smart_weight")
            self.equal_weights = None  # default equal weights to None, will be set later if needed
