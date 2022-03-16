# handle circular imports that result from typehinting
from __future__ import annotations

# queue, stack, and priority queue
from heapq import heappush, heappop
from collections import deque
import bisect

# scientific and graph libraries
import numpy as np
import networkx as nx
from strawberryfields.apps.clique import c_0 as sf_C0
from networkx.algorithms.approximation import max_clique as get_max_clique

# typehinting
# TYPE_CHECKING is set to False at runtime, but is True for the type hinting pass
# conditional import resolve circular import case caused by type hinting
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from explainers.facet_bb import FACETBranchBound

ordering = ""  # global for what ordering to use for the Branch and Bound tree exploration


class BBNode():
    '''
    A class for representing nodes in the branch and bound search tree

    Throughout we use type hinting `variable : type = value` to indicate the expected type of variable, this helps with readability and IDE code completion suggestions
    '''

    def __init__(self, parent, vertex: int, children=[]):
        '''
        Parameters
        ----------
        parent : branch and bound node
        vertex : the integer index of the graph vertex from the adjancency matrix
        children : an optional list of branch and bound nodes
        '''
        self.parent: BBNode = parent
        self.children: list[BBNode] = children
        self.partial_example = None
        self.lower_bound = np.inf
        self.new_vertex = vertex
        self.clique_candidates = None

        if parent:
            self.clique = self.parent.clique.copy()  # starting from parent clique
            bisect.insort(self.clique, vertex)  # insert the new vertex, keeping the list sorted
        else:
            self.clique = []  # if parent is None, this is the root node, start with empty clique

    # def __del__(self):
    #     # print("destructed")
    #     pass

    def __lt__(self, other: BBNode):
        '''
        In heapq algorithm, if two items have the same priority they are compared directly to determine which one should be higher in the priority queue. In this case if we have two nodes representing cliques with equal distances size we favor cliques which are closer to the majority size.
        '''
        global ordering
        if ordering == "ModifiedPriorityQueue":
            return self.lower_bound < other.lower_bound
        else:
            return len(self.clique) > len(other.clique)

    def find_candidates(self, G: nx.Graph):
        # Find C0: the set of all vertices in G which are sythesizaeble with the current clique
        self.clique_candidates: list[int] = []

        # TODO: When looking for C0 of this node, it must be a strict subset of its parents C0, attempted to shrink the adjacency matrix to consider, but need to reindex the resulting array which is smaller. Essentially would need a data structure which maps between the original adjacency indexs and the shrunken adjacency indexs
        # a = adjacency
        # if self.parent:
        #     # C0 for this node is a strict subset of C0 for the parent, reduces # of vertices to check
        #     idxs = self.parent.clique_candidates
        #     a = a[idxs][:, idxs]  # keep only the rows and colums corresponding to C0 of the parent
        # g = nx.Graph(a)

        self.clique_candidates = sf_C0(self.clique, G)  # returns all nodes when self.clique = []


class BranchBound():
    def __init__(self, explainer: FACETBranchBound, hyperparameters: dict):
        self.explainer = explainer
        self.majority_size = explainer.majority_size
        self.nlucky_guesses = 0  # how often is the heuristic the optimal solution
        self.nextensions = []
        self.nlower_bounds = []
        self.best_times = []
        self.hyperparameters = hyperparameters

        # parse bounding methods to use
        if hyperparameters.get("bb_upperbound") is None:
            print("No bb_upperbound set, defaulting to lower only")
            self.use_upper_bound = False
        else:
            self.use_upper_bound = hyperparameters.get("bb_upperbound")

        # parse ordering datastructure type to use
        global ordering
        if hyperparameters.get("bb_ordering") is None:
            print("No bb_ordering set, defaulting to PriorityQueue")
            self.ordering = "PriorityQueue"
            ordering = "PriorityQueue"
        else:
            self.ordering = hyperparameters.get("bb_ordering")
            ordering = hyperparameters.get("bb_ordering")
            if self.ordering not in ["PriorityQueue", "Stack", "Queue", "ModifiedPriorityQueue"]:
                print("invalid bb_ordering, using PriorityQueue")
                self.ordering = "PriorityQueue"
                ordering = "PriorityQueue"

        # parse ordering datastructure type to use
        hard_voting = hyperparameters.get("rf_hardvoting")
        if hard_voting is None or hard_voting is False:
            self.double_check = True
        else:
            self.double_check = False

        # distance logging
        log_dist = hyperparameters.get("bb_logdists")
        if log_dist is not None:
            self.log_dist = log_dist
        else:
            self.log_dist = False

        if self.log_dist:
            self.intermediate_dists = []

    def initial_guess(self, instance: np.ndarray, desired_label: int) -> np.ndarray:
        '''
        A heuristic for selecting a first counterfactual example

        Creates a counterfactual example from a random majority size subset of paths from the desired class's maxclique
        '''
        # initialize the counterfactual as a copy of the instance
        example = instance.copy()
        # the set of sythesizeable paths for the predicted class
        max_clique = self.explainer.max_cliques[desired_label]
        # randomly pick a set of paths to merge, stored as index pairs [[tree_id, path_id]]
        # merge_paths_idxs = max_clique[np.random.choice(len(max_clique), size=self.majority_size, replace=False)]
        # !WARNING: currently using entire max clique rather than nmajority of max clique to resolve soft voting
        merge_paths_idxs = max_clique
        # determine the bounds of the hyper-rectangle formed by those paths
        feature_bounds = self.explainer.compute_syth_thresholds(merge_paths_idxs, example.shape[0])
        # sythesize a counterfactual example which falls within those bounds with minimum distance from x
        example = self.explainer.fit_thresholds(example, feature_bounds)
        return example

    def branch(self, node: BBNode, instance: np.ndarray, desired_label: int):
        if self.use_upper_bound:
            self.branch_lower_upper(node, instance, desired_label)
        else:
            self.branch_lower(node, instance, desired_label)

    def branch_lower(self, node: BBNode, instance: np.ndarray, desired_label: int):
        # determine C0 the set of vertices which are sythesizable with C
        node.find_candidates(self.explainer.graphs[desired_label])

        # if there are sufficient vertices to reach majority size
        if self.solution_possible(node):
            # create node for each clique candidate
            for vertex in node.clique_candidates:
                child = BBNode(parent=node, vertex=vertex)
                # check that we haven't have visited this clique before
                key = hash(tuple(child.clique))
                if not key in self.clique_visited:
                    # remember we have visited the clique
                    self.clique_visited[key] = True
                    # compute lower bound for the node
                    child.partial_example, child.lower_bound = self.lower_bound(child, instance, desired_label)
                    # if the resulting nodes has a better lower bound than the current best, add it to the tree
                    if(child.lower_bound < self.best_distance):
                        node.children.append(child)
                        self.enqueue(child)

    def branch_lower_upper(self, node: BBNode, instance: np.ndarray, desired_label: int):
        # determine C0 the set of vertices which are sythesizable with C
        if node.clique_candidates is None:
            node.find_candidates(self.explainer.graphs[desired_label])

        # if there are sufficient vertices to reach majority size
        # if node.solution_possible(self.majority_size):

        # create node for each clique candidate
        for vertex in node.clique_candidates:
            child = BBNode(parent=node, vertex=vertex)
            # check that we haven't have visited this clique before
            key = hash(tuple(child.clique))
            if not key in self.clique_visited:
                # remember we have visited the clique
                self.clique_visited[key] = True

                # check if the child can have a solution
                child.find_candidates(self.explainer.graphs[desired_label])
                if self.solution_possible(child):
                    # compute the best possible distance of a solution which could exist
                    child.partial_example, child.lower_bound = self.lower_bound(child, instance, desired_label)
                    # if the possible solution could be better than the current solution
                    if(child.lower_bound < self.best_distance):
                        # look ahead and attempt to find a solution along the branch
                        pessimistic_example, upper_bound = self.upper_bound(child, instance, desired_label)
                        # if the lookahead solution actually exists
                        if(pessimistic_example is not None):
                            # if the lookahead solution is good, keep it
                            if upper_bound < self.best_distance:
                                self.set_best_solution(example=pessimistic_example, distance=upper_bound)
                            # if the child could contain a better solution than the current best, enqueue it
                            if(child.lower_bound < self.best_distance):
                                node.children.append(child)
                                self.enqueue(child)

    def upper_bound(self, node: BBNode, instance: np.ndarray, desired_label: int) -> Tuple[np.ndarray, float]:
        # determine C0 the set of vertices which are sythesizable with C
        if node.clique_candidates is None:
            node.find_candidates(self.explainer.graphs[desired_label])

        # get the list of all vertices which can be inherited from this node
        possible_vertices = node.clique + node.clique_candidates

        # select on the adjacency rows which contain these vertices
        adjacency = self.explainer.adjacencys[desired_label]
        adjacency = adjacency[possible_vertices][:, possible_vertices]

        # treat this as a graph and find the max clique
        g = nx.Graph(adjacency)
        max_clique_ixds = get_max_clique(g)
        max_clique_vertices = []
        for idx in max_clique_ixds:
            max_clique_vertices.append(possible_vertices[idx])

        # build a counterfactual example in the max clique as a worst case
        tree_paths = []
        for idx in max_clique_vertices:
            tree_paths.append(self.explainer.idx_to_treepath[desired_label][idx])
        example = instance.copy()
        feature_bounds = self.explainer.compute_syth_thresholds(tree_paths, example.shape[0])
        example = self.explainer.fit_thresholds(example, feature_bounds)
        upper_bound = self.explainer.distance_fn(instance, example)

        # if the max clique is of insufficient size, the example may not be counterfactual
        if self.double_check:
            if not self.check_example(example):
                example = None
                upper_bound = np.inf
        else:
            if len(tree_paths) < self.majority_size:
                example = None
                upper_bound = np.inf

        return example, upper_bound

    def lower_bound(self, node: BBNode, instance: np.ndarray, desired_label: int) -> Tuple[np.ndarray, float]:
        self.nlower_bounds[-1] += 1
        # convert the path indexs to path ids
        tree_paths = []
        for idx in node.clique:
            tree_paths.append(self.explainer.idx_to_treepath[desired_label][idx])

        # TODO improve performance of feature bounding by having bounds be added recursively relative to the parent, rather than build from scratch each time
        # build the counterfactual example from this clique
        example = node.parent.partial_example.copy()
        feature_bounds = self.explainer.compute_syth_thresholds(tree_paths, example.shape[0])
        example = self.explainer.fit_thresholds(example, feature_bounds)

        lower_bound = self.explainer.distance_fn(instance, example)
        return example, lower_bound

    def check_example(self, example: np.ndarray, desired_label: int) -> bool:
        return self.explainer.model.predict([example]) == desired_label

    def solution_possible(self, node: BBNode) -> bool:
        # !WARNING: only works with hard voting
        return (len(node.clique) + len(node.clique_candidates)) >= self.majority_size

    def is_solution(self, node: BBNode, instance: np.ndarray, desired_label: int) -> bool:
        if self.double_check:
            return self.check_example(node.partial_example, desired_label)
        if not self.double_check:
            return len(node.clique) >= self.majority_size

    def set_best_solution(self, example: np.ndarray, distance):
        self.best_solution = example
        self.best_distance = distance
        # track the number of nodes bounded before the optimal solution is found
        self.best_times[-1] = self.nlower_bounds[-1]
        if self.log_dist:
            self.intermediate_dists[-1].append([self.best_times[-1], self.best_distance])
        # print("New Best {:d}, {:.6f}".format(self.best_times[-1], self.best_distance))

    def enqueue(self, node: BBNode):
        self.nextensions[-1] += 1
        if self.ordering == "PriorityQueue":
            heappush(self.queue, (node.lower_bound, node))
        elif self.ordering == "ModifiedPriorityQueue":
            priority = self.majority_size - len(node.clique)
            heappush(self.queue, (priority, node))
        elif self.ordering == "Stack":
            self.queue.append(node)
        elif self.ordering == "Queue":
            self.queue.append(node)

    def dequeue(self) -> BBNode:
        if self.ordering == "PriorityQueue":
            return heappop(self.queue)[1]  # heappop yields (priority, node)
        elif self.ordering == "ModifiedPriorityQueue":
            return heappop(self.queue)[1]  # heappop yields (priority, node)
        elif self.ordering == "Stack":
            return self.queue.pop()
        elif self.ordering == "Queue":
            return self.queue.popleft()

    def init_queue(self):
        if self.ordering == "PriorityQueue":
            self.queue = []
        elif self.ordering == "ModifiedPriorityQueue":
            self.queue = []
        elif self.ordering == "Stack":
            self.queue = deque()
        elif self.ordering == "Queue":
            self.queue = deque()

    def solve(self, instance: np.ndarray, desired_label: int) -> np.ndarray:
        self.nextensions.append(0)
        self.nlower_bounds.append(0)
        self.best_times.append(0)
        if self.log_dist:
            self.intermediate_dists.append([])

        # initialize the first guess using a heuristic method
        initial_guess = self.initial_guess(instance, desired_label)
        initial_dist = self.explainer.distance_fn(instance, initial_guess)
        self.set_best_solution(example=initial_guess, distance=initial_dist)

        self.clique_visited = {}

        # create a root node and add it to the queue
        self.init_queue()
        root = BBNode(parent=None, vertex=-1, children=[])
        root.partial_example = instance.copy()
        root.lower_bound = 0
        self.enqueue(root)

        while len(self.queue) > 0:  # while there is nodes in the queue
            node: BBNode = self.dequeue()  # fetch the next node
            # if the current node could be better than the best guess, examine it
            if node.lower_bound < self.best_distance:
                # if node is a solution, its partial example is counterfactual
                if self.is_solution(node, instance, desired_label):
                    # update the current best solution
                    self.set_best_solution(node.partial_example, node.lower_bound)
                else:  # if the node is not a solution, expand it by branching
                    self.branch(node, instance, desired_label)
            # if the current node can't be better than the current best
            else:
                # And we're use priority queue
                if self.ordering == "PriorityQueue":
                    # all remaining enqueued nodes are worse than this one, terminate
                    return self.best_solution
                # otherwise continue to the next node

        if(self.best_solution == initial_guess).all():
            self.nlucky_guesses += 1
        return self.best_solution
