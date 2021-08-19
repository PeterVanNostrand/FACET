from numpy.core.numeric import full
from explainers.explainer import Explainer
from sklearn.ensemble import IsolationForest as skIsolationForest
import numpy as np

import networkx as nx
from networkx.algorithms.approximation import clique

from utilities.metrics import dist_euclidean
from utilities.metrics import dist_features_changed


class GraphMerge(Explainer):
    def __init__(self, model, hyperparameters=None):
        self.model = model
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

    def explain(self, x, y):
        '''
        A method for explaining the samples in x by finding the best candidate contrastive examples generated by the model's detectors

        Parameters
        ----------
        x               : an array of samples, dimensions (nsamples, nfeatures)
        y               : an array of labels which correspond to the labels, (nsamples, )

        Returns
        -------
        best_examples : an array of contrastive examples with dimensions (nsamples, nfeatures). Each of final_examples[i] corresponds to
        the best examples which explains x[i] from those suggested by the detectors
        '''

        # !WARNING : This method only defined for an ensemble containing only a single random forest detector

        # build a graph to represent the feature similarities between trees
        rf_detector = self.model.detectors[0]
        adjacency = rf_detector.get_tree_adjacency()
        adjacency = np.floor(adjacency)  # consider only fully disjoint trees for merging

        # create a graph from the adjacency matrix using networkx
        G = nx.Graph(adjacency)

        # identify the largest set of trees which are fully disjoint in the features they use this is done by finding the largest complete (i.e. fully connectected) subgraph
        # returns a set of node indices representing trees in the forest
        fully_disjoint_trees = list(clique.max_clique(G))

        # keep enough disjoint trees to form a majority
        n_majority = (int)(np.floor(rf_detector.ntrees / 2) + 1)
        fully_disjoint_trees = fully_disjoint_trees[:n_majority]

        xprime = x.copy()  # an array for the constructed contrastive examples

        # get a candidate contrastive example for each sample from each of the disjoint trees
        candidates = rf_detector.get_candidate_examples_treewise(x, y, fully_disjoint_trees)

        i = 0
        for i in range(len(fully_disjoint_trees)):
            tree_id = fully_disjoint_trees[i]

            # merge the examples from each tree into xprime, taking only the features which that tree used
            features_used = rf_detector.get_features_used(tree_id)
            xprime[:, features_used] = candidates[i, :, features_used].T
            i += 1

        return xprime
