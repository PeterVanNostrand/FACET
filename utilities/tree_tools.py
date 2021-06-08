
import numpy as np
from utilities.metrics import euclidean_distance


class TreeContraster():
    def __init__(self, tree, verbose=False, feature_names=None, class_names=None,):
        self.t = tree
        self.fnames = feature_names
        self.cnames = class_names
        self.verbose = verbose

        self.all_paths = []
        self.__in_order_path(self.all_paths)

    def __in_order_path(self, built_paths=[], node_id=0, path=[], path_string=""):
        '''
        An algorithm for pre-order binary tree traversal. This walks throuhg the entire tree generating a list of paths from the root node to a leaf
        and recording the final classification of the leaf node for that path. Paths are stored as `p = [f, g, h, i, j]` where each letter reprsents the
        node_id taken in that path, with `f` being the root node and `j` being the leaf node

        Parameters
        ----------
        t           : the decision tree classifier to travers
        built_paths : the return values, a list of tuples (`class_id` = integer class ID, `path=[f, g, h, i, j]`)
        node_id     : the `node_id` to start traversal at, defaults to root node of tree (`node_id=0`)
        path        : the starting path up to by not including the node referenced by `node_id`, defaults to empty
        path_string : a running text explanation of the features and values used to split along the path
        verbose     : when true prints `path_string` during execution, `default=False`

        Returns
        -------
        None : see the output parameter `built_paths`
        '''
        # process current node
        path = path.copy()
        path.append(node_id)

        if self.verbose:
            if self.feature[node_id] >= 0:  # if its a split node, get the attribute and value
                fname = self.feature_names[self.t.tree_.feature[node_id]]
                fvalue = self.t.threshold[node_id]
                path_component = "{fname} <= {fvalue:0.2f}, ".format(fname=fname, fvalue=fvalue)
            else:  # otherwise its a leaf node, determine the class
                cid = np.argmax(self.t.value[node_id])
                cname = self.class_names[cid]
                path_component = "leaf: " + cname
            path_string = path_string + path_component

        # If the node is a leaf
        if(self.t.tree_.children_left[node_id] == -1 and self.t.tree_.children_right[node_id] == -1):
            if self.verbose:
                print(path)
                print(path_string)
            cid = np.argmax(self.t.tree_.value[node_id])
            built_paths.append((cid, path))
            return

        # process left child
        self.__in_order_path(built_paths, self.t.tree_.children_left[node_id], path, path_string)

        # process right node
        self.__in_order_path(built_paths, self.t.tree_.children_right[node_id], path, path_string)

    def contrast_instance(self, instance, label, difference=0.01):
        '''
        Performs contrastive example generation for a single sample using the tree associated with the TreeHelper instance

        Parameters
        ----------
        instance   : the given sample to perform contrastive example generation on
        label      : the class label associated with the given instance
        difference : the percent change relative of the threshold value to offest to each feature to "nudge" the contrastive example past thresholds

        Returns
        -------
        instance_examples : an array of minimally modified versions of the instance, one for each path from the root node to a leaf
                            with a class other than that of the label
        '''
        # initialize empty output array of contrastive examples
        instance_examples = []

        # for every possible path from root to leaf
        for path_class, path in self.all_paths:
            # if the result of that path is a label the same as x, skip it
            if path_class == label:
                continue

            # print the path for the current example being generated
            if self.verbose:
                print(path_class, path)

            # initialize a constastive example as x
            xprime = instance.copy()

            # following the given path up to (but not including) the leaf node
            for i in range(len(path)-1):
                node_id = path[i]           # the current node in the path
                next_node_id = path[i+1]    # the next node in the path

                fid = self.t.tree_.feature[node_id]          # get the feature the current node splits on
                fthreshold = self.t.tree_.threshold[node_id]  # get the value the current node splits on

                # if the path goes left and x went right, force xprime left
                if (next_node_id == self.t.tree_.children_left[node_id] and xprime[fid] > fthreshold):
                    xprime[fid] = fthreshold - np.abs(difference * fthreshold)

                # if the path goes right and x went left, force xprime right
                if (next_node_id == self.t.tree_.children_right[node_id] and xprime[fid] <= fthreshold):
                    xprime[fid] = fthreshold + np.abs(difference * fthreshold)

            # save the example
            instance_examples.append(xprime)

        return np.array(instance_examples)

    def construct_examples(self, x, y, difference=0.01, distance_metric="Euclidean"):
        '''
        A function to contruct a set of contrastive examples for the tree `t` such that each example has a predicted class different than that of `x`.
        This is done by making minimal modifications to x such that it follows a path from the root of `t` to a leaf node with a predicted class other
        than that of `y`

        Parameters
        ----------
        t          : the decision tree to be used in contrastive example generation
        x          : an array of samples to generate contrastive examples for
        y          : an array of labels represeinting the classes of the given samples in `x`
        difference : the percent change relative of the threshold value to offest to each feature to "nudge" the contrastive example past thresholds
        verbose    : when true prints the path for each example generated and the expected class of that paths leaf node

        Returns
        -------
        examples   : an array of contrastive examples, one for each sample in 'x'. The example for each instance in `x` is chosen using the given distance_metric
        '''

        # select the distance function which corresonds to the provided distance metric
        if distance_metric == "Euclidean":
            distance_fn = euclidean_distance
        else:
            print("Unknown distance function {}, using Euclidean distance".format(distance_metric))
            distance_fn = euclidean_distance

        examples = np.empty(shape=x.shape)  # output array for generated examples
        for i in range(x.shape[0]):  # for each instance in the array x
            instance = x[i]         # get the instance
            label = y[i]            # and its corresponding label

            # generate contrastive examples for that instance
            instance_examples = self.contrast_instance(instance, label, difference)
            # compute the distance between each instance and its examples
            distances = distance_fn(instance, instance_examples)
            # find the most similar example according to the distance function
            minimal_index = np.argmin(distances)
            # and select that as the example for this instance
            examples[i] = instance_examples[minimal_index]

        return examples
