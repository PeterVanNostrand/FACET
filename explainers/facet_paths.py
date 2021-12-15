import math
from os import replace
from networkx.readwrite.json_graph import adjacency
from explainers.explainer import Explainer
import numpy as np

import networkx as nx
from networkx.algorithms.approximation import max_clique as get_max_clique

from utilities.metrics import dist_euclidean
from utilities.metrics import dist_features_changed

import matplotlib.pyplot as plt
from sklearn import tree
from itertools import combinations


class FACETPaths(Explainer):
    def __init__(self, model, hyperparameters=None):
        self.model = model
        self.parse_hyperparameters(hyperparameters)

    def prepare(self):
        rf_detector = self.model.detectors[0]
        rf_trees = rf_detector.model.estimators_
        rf_ntrees = len(rf_trees)
        rf_nclasses = rf_detector.model.n_classes_
        rf_nfeatures = len(rf_detector.model.feature_importances_)

        self.build_paths(rf_trees)
        self.index_paths()
        self.find_synthesizeable_paths(rf_trees)
        self.build_graph(rf_trees, rf_nclasses)

    def build_graph(self, trees, nclasses):
        adjacencys = self.build_adjacencys(trees, nclasses)
        graphs = []
        max_cliques = []
        for i in range(nclasses):
            g = nx.Graph(adjacencys[i])

            clique_idxs = list(get_max_clique(g))
            clique_paths = []
            for idx in clique_idxs:
                clique_paths.append(self.idx_to_treepath[idx])
            clique_paths = np.array(clique_paths)
            clique_paths = clique_paths[clique_paths[:, 0].argsort()]

            graphs.append(g)
            max_cliques.append(clique_paths)

        self.graphs = graphs
        self.max_cliques = max_cliques

        if(not self.check_cliques(max_cliques)):
            print("BAD CLIQUE")
        print("Clique Sizes:")
        for clique in max_cliques:
            print("\t {}".format(len(clique)))

    def build_adjacencys(self, trees, nclasses):
        ntrees = len(trees)
        adjacencys = np.zeros(shape=(nclasses, self.npaths, self.npaths))

        for t1_id in range(ntrees):
            for t2_id in range(ntrees):
                t1_t2_merges = self.sythesizable_paths[t1_id][t2_id]
                for p1_id in range(len(t1_t2_merges)):
                    t1p1_index = self.treepath_to_idx[t1_id][p1_id]
                    t1p1_class = int(self.all_paths[t1_id][p1_id][-1:, 3:][0][0])
                    for p2_id in t1_t2_merges[p1_id]:
                        t2p2_index = self.treepath_to_idx[t2_id][p2_id]
                        adjacencys[t1p1_class][t1p1_index][t2p2_index] = 1

        return adjacencys

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

    def index_paths(self):
        ntrees = len(self.all_paths)
        treepath_to_idx = []
        idx_to_treepath = []

        index = 0
        for i in range(ntrees):
            treei_to_idx = []
            for j in range(len(self.all_paths[i])):
                treei_to_idx.append(index)
                idx_to_treepath.append((i, j))
                index += 1
            treepath_to_idx.append(treei_to_idx)

        print("Num paths: {}".format(index))

        self.npaths = index
        self.treepath_to_idx = treepath_to_idx
        self.idx_to_treepath = idx_to_treepath

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

    def check_cliques(self, cliques):
        '''
        Returns true iff for each clique, every combination of paths within the clique is synthesizeable
        '''
        all_synthesizeable = True
        for clique in cliques:
            for i in range(len(clique)):
                t1_id = clique[i][0]
                p1_id = clique[i][1]
                p1 = self.all_paths[t1_id][p1_id]
                for j in range(i+1, len(clique)):
                    t2_id = clique[j][0]
                    p2_id = clique[j][1]
                    p2 = self.all_paths[t2_id][p2_id]
                    if not self.is_mergable(p1, p2):
                        all_synthesizeable = False
        return all_synthesizeable

    def share_feature(self, p1, p2, fi):
        '''
        Returns true if the two paths both use feature_i in at least one node, false otherwise.

        Parameters
        ----------
        p1, p2: array results from __in_order_path
        fi: int index of feature i
        '''
        p1_features = p1[:, 1:2]
        p2_features = p2[:, 1:2]

        return (fi in p1_features) and (fi in p2_features)

    def same_outcome(self, p1, p2):
        '''
        Returns true if and only if p1 and p2 lead to leaf nodes of the same class.
        '''
        p1_pred = p1[-1:, -1:][0][0]
        p2_pred = p2[-1:, -1:][0][0]

        return (p1_pred == p2_pred)

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

    def save_tree_figs(self, t1, t2, path="C:/Users/Peter/Downloads/"):
        tree.plot_tree(t1)
        plt.savefig(path + "t1.png")
        tree.plot_tree(t2)
        plt.savefig(path + "t2.png")

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

    def explain(self, x, y):
        '''
        Parameters
        ----------
        x               : an array of samples, dimensions (nsamples, nfeatures)
        y               : an array of predicted labels which correspond to the labels, (nsamples, )

        Returns
        -------
        xprime : an array of contrastive examples with dimensions (nsamples, nfeatures)
        '''
        xprime = x.copy()  # an array for the constructed contrastive examples

        # the number of trees which much aggree for a prediction
        majority_size = math.floor(len(self.all_paths) / 2) + 1

        # assumimg binary classification [0, 1] set counterfactual class
        counterfactual_class = ((y - 1) * -1)

        for i in range(x.shape[0]):
            counter_label = counterfactual_class[i]  # the predicted label to be explained for this instance
            clique = self.max_cliques[counter_label]  # the set of sythesizeable paths for the predicted class

            if self.mode == "fast":
                # randomly pick a set of paths to merge, stored as index pairs [[tree_id, path_id]]
                merge_paths_idxs = clique[np.random.choice(len(clique), size=majority_size, replace=False)]
                feature_bounds = self.compute_syth_thresholds(merge_paths_idxs, x.shape[1])
                xprime[i] = self.fit_thresholds(xprime[i], feature_bounds)
            elif self.mode == "exhaustive":
                # exhausively test every possible set of paths to merge, keeping the closest explanation
                min_distance = float("inf")
                all_sets = combinations(clique, majority_size)
                for path_set in all_sets:
                    feature_bounds = self.compute_syth_thresholds(path_set, x.shape[1])
                    xprime_candidate = self.fit_thresholds(xprime[i].copy(), feature_bounds)
                    cand_distance = self.distance_fn(x[i], xprime_candidate)
                    if cand_distance < min_distance:
                        xprime[i] = xprime_candidate
                        min_distance = cand_distance

        # check that all counterfactuals result in a different class
        preds = self.model.predict(xprime)
        failed_explanation = (preds == y)
        xprime[failed_explanation] = np.tile(np.inf, x.shape[1])

        return xprime

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

    def get_clique_size(self):
        total_members = 0
        for clique in self.max_cliques:
            total_members += len(clique)

        return total_members / len(self.max_cliques)

    def parse_hyperparameters(self, hyperparameters):
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

        # greedy sythesis
        if hyperparameters.get("expl_greedy") is None:
            print("No expl_greedy set, defaulting to False")
            self.greedy = False
        else:
            self.greedy = hyperparameters.get("expl_greedy")

        # threshold offest for picking new values
        offset = hyperparameters.get("facet_offset")
        if offset is None:
            print("No facet_offset provided, using 0.01")
            self.offset = 0.01
        else:
            self.offset = offset

        # explanation mode: fast or exhaustive
        mode = hyperparameters.get("facet_mode")
        if mode is None:
            print("No facet_mode provided, using exhaustive")
            self.mode = "exhaustive"
        else:
            self.mode = mode
