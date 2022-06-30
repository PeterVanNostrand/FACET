# handle circular imports that result from typehinting
from __future__ import annotations
import bisect

import numpy as np
from bitarray import bitarray
from bitarray.util import zeros as bitzeros

from typing import TYPE_CHECKING

from pydantic import NoneIsAllowedError, constr
if TYPE_CHECKING:  # circular import avoidance
    from explainers.facet_index import FACETIndex

LOWER = 0
UPPER = 1


class BitVectorIndex():
    '''
    A method for performing high dimensional indexing of hyper-rectangles using a set of precomputed redundant bit vectors. Designed to efficiently find the nearest hyperrectangle to a point subject to an optional set of constraints along each axis.

    Based on "Indexing High Dimensional Rectangles for Fast Multimedia Identification" by Jonathan Goldstein, John Platt, Christopher Burges. 2003 Microsoft Research tecnical report
    '''

    def __init__(self, rects: list[np.ndarray], explainer: FACETIndex, hyperparameters: dict):
        '''
        Parameters
        ----------
        rects: the list of hyperrectangle records to index, all records should be of the same class
        m: the nuymber of intervals to create along each dimension
        '''
        self.parse_hyperparameters(hyperparameters)
        self.explainer = explainer
        # combine the list of hyperrectangles into one array of shape (nrects, ndim, 2)
        self.rects: np.ndarray = np.stack(rects, axis=0)
        self.nrects = self.rects.shape[0]
        self.ndimensions = self.rects.shape[1]

        # select the partition values for each interval
        self.intervals, self.indexed_dimensions = self.generate_intervals(self.rects)
        if self.verbose:
            print("N Indexed Dimensions:", sum(self.indexed_dimensions))
        self.rbv = self.build_bit_vectors(self.rects)
        self.search_log = []  # for experiments store the # of rects search for each sample explained

    def point_query(self, instance: np.ndarray, constraints: np.ndarray = None, weights: np.ndarray = None) -> np.ndarray:
        '''
        Uses the bit vector index to find hyper-rectangles which are close to the given point

        Parameters
        ----------
        instance: a numpy array of shape (ndim,) to search around
        constraints : a numpy array of shape (ndim, 2) representing the user's constrained min/max values along each axis
        weights: a numpy array of shape (ndim) representing the user's willingness to change each feature with higher weight indicating more willing to change

        Returns
        -------
        closest-rect: a numpy array of shape(ndim, 2) of the nearest hyper-rectangle that matches the user constratints
        '''
        # create a hyper-sphere around the point and indentify which intervals it covers. We do this by creating a hyper-sphere with the initial radius, converting it to a hyper-rectangle, and searching for records in that rect
        closest_rect = None
        closest_dist = np.inf
        solution_found = False
        search_complete = False  # we have searched the entire constraint range

        # bit vector for the rects we have already checked the distance to
        searched_bits = bitzeros(self.nrects)
        search_radius = self.initial_radius
        while not solution_found and not search_complete:
            # convert the query hypersphere into a hyperrectangle
            query_rect = np.zeros(shape=(self.ndimensions, 2))
            if weights is not None:
                query_rect[:, LOWER] = (instance - weights * search_radius)
                query_rect[:, UPPER] = (instance + weights * search_radius)
            else:
                query_rect[:, LOWER] = (instance - search_radius)
                query_rect[:, UPPER] = (instance + search_radius)
            nonzero_query_area = True

            # if applicable restrict the search radius to the user provided constraints
            if constraints is not None:
                if have_intersection(query_rect, constraints):
                    # take intersection of query_rect and constraints if possible
                    query_rect[:, LOWER] = np.maximum(query_rect[:, LOWER], constraints[:, LOWER])  # raise lower bounds
                    query_rect[:, UPPER] = np.minimum(query_rect[:, UPPER], constraints[:, UPPER])  # lower upper bounds
                    search_complete = (query_rect == constraints).all()  # search are encloses entire constrained region
                else:
                    nonzero_query_area = False

            if nonzero_query_area:
                # get the set of hyper-rect records in the query rectangle
                matching_bits = self.rect_query(query_rect)
                # exclude rectangles which we have already checked the distance to
                new_match_bits = (matching_bits & ~searched_bits)

                # if we have new matches, check their distance
                if new_match_bits.any():
                    # record the new matches as searched
                    searched_bits |= new_match_bits
                    # expand the packed bitarry to an array of booleans
                    new_match_slice = np.array(new_match_bits.tolist(), dtype=bool)
                    # get the matching rectangles
                    new_rects = self.rects[new_match_slice]
                    # search the matching rects for the nearest rectangle within the search radius. Its possible that the matching set is non-empty due to a rectangle in an unindexed dimension that is further than the search radius, which is not guaranteed to be the nearest to the point
                    for rect in new_rects:
                        # if applicable only consider the rectangle if it falls within the constraints
                        if constraints is None or have_intersection(rect, constraints):
                            if constraints is not None:  # take only part of rect which falls in constraints
                                rect[:, LOWER] = np.maximum(rect[:, LOWER], constraints[:, LOWER])  # raise lower bounds
                                rect[:, UPPER] = np.minimum(rect[:, UPPER], constraints[:, UPPER])  # lower upper bounds
                            test_instance = self.explainer.fit_to_rectangle(instance, rect)
                            dist = self.explainer.distance_fn(instance, test_instance, weights)
                            # valid solutions must fall within the search radius
                            if dist < closest_dist:
                                closest_rect = rect
                                closest_dist = dist

                # if the best solution falls within the search radius, exit
                solution_found = (closest_dist <= search_radius)
                # if we've searched the whole constraints region and found no solution
                if search_complete and not solution_found:
                    closest_rect = None  # return Null

            if self.radius_growth == "Linear":
                search_radius += self.initial_radius
            elif self.radius_growth == "Exponential":
                search_radius *= 2
        # Experiment logging
        nrects_searched = searched_bits.count()
        self.search_log.append(nrects_searched)
        if self.verbose:
            print(nrects_searched)

        return closest_rect

    def k_point_query(self, instance: np.ndarray, constraints: np.ndarray = None, weights: np.ndarray = None, k: int = 1) -> np.ndarray:
        '''
        Uses the bit vector index to find hyper-rectangles which are close to the given point

        Parameters
        ----------
        instance: a numpy array of shape (ndim,) to search around
        constraints : a numpy array of shape (ndim, 2) representing the user's constrained min/max values along each axis
        weights: a numpy array of shape (ndim) representing the user's willingness to change each feature with higher weight indicating more willing to change
        k: int value >=1 corresponding the number of nearest counterfactuals to return

        Returns
        -------
        k_closeset_rects: a list of arrays of shape(nrecords, ndim, 2) of the k nearest hyper-rectangles if found
        '''
        # create a hyper-sphere around the point and indentify which intervals it covers. We do this by creating a hyper-sphere with the initial radius, converting it to a hyper-rectangle, and searching for records in that rect
        rect_dists = []  # a priority queue of the dist to each searched hyper-rect, ordered by increasing distance
        trimmed_rects = {}  # a dict rect_id -> bool. True iff the rect was trimmed to fit in the constraints region
        solution_found = False
        search_complete = False  # we have searched the entire constraint range

        # bit vector for the rects we have already checked the distance to
        rect_ids = np.array(range(self.nrects))  # id each rect by its location in the enumerated list
        searched_bits = bitzeros(self.nrects)  # keep track of which rects we've already checked
        search_radius = self.initial_radius
        while not solution_found and not search_complete:
            # convert the query hypersphere into a hyperrectangle
            query_rect = np.zeros(shape=(self.ndimensions, 2))
            if weights is not None:
                query_rect[:, LOWER] = (instance - weights * search_radius)
                query_rect[:, UPPER] = (instance + weights * search_radius)
            else:
                query_rect[:, LOWER] = (instance - search_radius)
                query_rect[:, UPPER] = (instance + search_radius)
            nonzero_query_area = True

            # if applicable restrict the search radius to the user provided constraints
            if constraints is not None:
                if have_intersection(query_rect, constraints):
                    # take intersection of query_rect and constraints if possible
                    query_rect[:, LOWER] = np.maximum(query_rect[:, LOWER], constraints[:, LOWER])  # raise lower bounds
                    query_rect[:, UPPER] = np.minimum(query_rect[:, UPPER], constraints[:, UPPER])  # lower upper bounds
                    search_complete = (query_rect == constraints).all()  # search are encloses entire constrained region
                else:
                    nonzero_query_area = False

            if nonzero_query_area:
                # get the set of hyper-rect records in the query rectangle
                matching_bits = self.rect_query(query_rect)
                # exclude rectangles which we have already checked the distance to
                new_match_bits = (matching_bits & ~searched_bits)

                # if we have new matches, check their distance
                if new_match_bits.any():
                    # record the new matches as searched
                    searched_bits |= new_match_bits
                    # expand the packed bitarry to an array of booleans
                    new_match_slice = np.array(new_match_bits.tolist(), dtype=bool)
                    # get the matching rectangles
                    new_rects = self.rects[new_match_slice]
                    new_rect_ids = rect_ids[new_match_slice]
                    # filter matching rects for those which fall within the constraints region and compute their dists
                    for rect, rect_id in zip(new_rects, new_rect_ids):
                        # if applicable only consider the rectangle if it falls within the constraints
                        if constraints is None or have_intersection(rect, constraints):
                            if constraints is not None:  # take only part of rect which falls in constraints
                                rect[:, LOWER] = np.maximum(rect[:, LOWER], constraints[:, LOWER])  # raise lower bounds
                                rect[:, UPPER] = np.minimum(rect[:, UPPER], constraints[:, UPPER])  # lower upper bounds
                                trimmed_rects[rect_id] = True
                            test_instance = self.explainer.fit_to_rectangle(instance, rect)
                            dist = self.explainer.distance_fn(instance, test_instance, weights)
                            # record the dist to this rect on the priority queue
                            bisect.insort(rect_dists, (dist, rect_id))

                # if the closest k rects fall within the search radius sufficent solutions were found, search complete
                solution_found = (len(rect_dists) >= k) and (rect_dists[k-1][0] <= search_radius)
            if self.radius_growth == "Linear":
                search_radius += self.initial_radius
            elif self.radius_growth == "Exponential":
                search_radius *= 2

        # return the top-k closest rects, returning an empty list if no rects were found
        k_closest_rects: list[np.ndarray] = []
        i = 0
        while i < k and i < len(rect_dists):
            dist, rect_id = rect_dists[i]
            # if only a portion of the rect falls in the constraints region
            rect = self.rects[rect_id].copy() # make copy to prevent modifying self.rects
            if constraints is not None and rect_id in trimmed_rects:
                rect[:, LOWER] = np.maximum(rect[:, LOWER], constraints[:, LOWER])  # raise lower bounds
                rect[:, UPPER] = np.minimum(rect[:, UPPER], constraints[:, UPPER])  # lower upper bounds
            k_closest_rects.append(rect)
            i += 1

        # Experiment logging
        nrects_searched = searched_bits.count()
        self.search_log.append(nrects_searched)
        if self.verbose:
            print(nrects_searched)

        return k_closest_rects

    def rect_query(self, query_rect: np.ndarray) -> bitarray:
        '''
        Finds the set of all record hyper-rectangles which overlap the region defined in the query rectangle

        Parameters
        ----------
        query_rect: a numpy array of shape (ndim, 2) representing the upper/lower bound along each axis to search in

        Returns
        -------
        matching_bits: a bitarray of length nrects with each bit set to one iff the corresponding hyper-rectangle fall in the query region
        '''
        # TODO rewrite using array slicing rather than iterative search?
        # start with the set of all record hyper-rectnalges
        matching_bits: bitarray = bitzeros(self.nrects)
        matching_bits.invert()
        # find the intervals which the upper and lower edges of the query rectangle fall into
        for dim in range(self.ndimensions):
            if self.indexed_dimensions[dim]:
                i = 0
                edges_found = 0
                while i < self.m and not edges_found == 2:
                    # get the upper and lower bounds of the interval range
                    lower_bound = self.intervals[dim][i][LOWER]
                    upper_bound = self.intervals[dim][i][UPPER]
                    # check if the query rectangles lower edge falls in this range
                    if query_rect[dim][LOWER] >= lower_bound and query_rect[dim][LOWER] < upper_bound:
                        # bitwise AND together to remove rectangles which fall below the lower edge of the query rect
                        matching_bits &= self.rbv[dim][LOWER][i]
                        edges_found += 1
                    # check if the query rectangles upper edge falls in this range
                    if query_rect[dim][UPPER] >= lower_bound and query_rect[dim][UPPER] < upper_bound:
                        # bitwise AND together to remove rectangles which fall above the upper edge of the query rect
                        matching_bits &= self.rbv[dim][UPPER][i]
                        edges_found += 1
                    i += 1
        return matching_bits

    def build_bit_vectors(self, rects: np.ndarray) -> list[list[list[bitarray]]]:
        '''
        Generates a redundant bit vector index for the given set of hyper-rectangle records

        Parameters
        ----------
        rects: a list of numpy arrays, each of shape (ndim, 2) corresponding to min/max values along each dimension

        Returns
        -------
        rbv: the bit vector index, a list of bitarrays of shape (ndim, 2, m), where rbv[i][UPPER/LOWER][j] corresponds to the bit vector index for the UPPER/LOWER bound for the jth interval of the ith dimension
        '''
        # Determine which rectangles are above the lower bounds and below the upper bounds for each interval. Given the below set of rectangles and the three intervals along the horizontal axis, we set the bit for the lower bound vector if the rectangle falls within in the given interval or any interval above it. We set the bit for the upper bound if the rectangle falls within the given interval or any interval below it
        #  | █  █████████  ████████ ██      |     P(I2,LB) P(I2,UB)
        #  | ████  █████  █████ ██████      |  R1     0       1
        #  ___________________________      |  R2     1       1
        #  |   I1   |   I2   |   I3   |     |  R3     1       1
        #                                   |  R4     1       0
        #  Lets number rects left to right  |  R5     0       1
        #  top to bottom, as below          |  R6     1       1
        #     R1, R2, R3, R4                |  R7     1       1
        #     R5, R6, R7, R8                |  R8     0        1

        # create the empty redudant bit vectors
        rbv = [[[bitarray() for _ in range(self.m)] for _ in range(2)] for _ in range(self.ndimensions)]
        # Dim LowerBoundVectors   UpperBoundVectors
        #  0  [P1L, P2L ... PML], [P1U, P2U ... PMU]
        #  1  [P1L, P2L ... PML], [P1U, P2U ... PMU]
        #          .......             .......
        #  D  [P1L, P2L ... PML], [P1U, P2U ... PMU]
        # rbv[dimension][lower/upper][interval]

        # a rectangle is above the lower bound for the interval if its upper edge along that axis is greater than or equal to the min value for the intervals range
        # i.e. if a hyper-rectangles edge falls on the boundary between two intervals, count it as in the rightmost (higher along the axis) of the two intervals
        for dim in range(self.ndimensions):  # for each dimension
            if self.indexed_dimensions[dim]:
                for i in range(self.m):  # for each of the m intervals
                    # build the lower bound and upper bound bit vectors
                    lb_bit_vec = [False for _ in range(self.nrects)]
                    ub_bit_vec = [False for _ in range(self.nrects)]
                    for j in range(self.nrects):  # check each rectangle
                        lb_bit_vec[j] = rects[j][dim][UPPER] >= self.intervals[dim][i][LOWER]  # r's upper edge above LB
                        ub_bit_vec[j] = rects[j][dim][LOWER] < self.intervals[dim][i][UPPER]  # r's lower edge below UB
                    # pack the booleans into bits of a word into using bitarray
                    rbv[dim][LOWER][i].extend(lb_bit_vec)
                    rbv[dim][UPPER][i].extend(ub_bit_vec)
        return rbv

    def generate_intervals(self, rects: np.ndarray):
        '''
        Generates a set of intervals based on the rectangles bound locations

        Returns
        -------
        intervals: an array of shape (ndim, m, 2) where intervals[i][j][0] represents the lower end of the range for interval j on dimension i and intervals[i][j][1] the upper
        '''
        indexed_dimensions = [True for _ in range(self.ndimensions)]
        intervals = np.zeros(shape=(self.ndimensions, self.m, 2))
        for dim in range(self.ndimensions):
            # get all the bounds for this dimension
            dim_bounds = rects[:, dim].flatten()
            # add +/- infinity as bounds if they are not already in array
            if not np.isin(dim_bounds, -np.inf).any():
                dim_bounds = np.append(dim_bounds, -np.inf)
            if not np.isin(dim_bounds, np.inf).any():
                dim_bounds = np.append(dim_bounds, np.inf)
            # deduplicate the set of bounds
            deduplicate = True
            if deduplicate:
                dim_bounds = np.unique(dim_bounds)
            # sort them in ascending order
            np.sort(dim_bounds)

            # if we have sufficient bounds distribute them evenly across the intervals
            if len(dim_bounds) >= self.m + 1:
                # determine the number of bounds per each interval
                group_size = len(dim_bounds) // (self.m + 1)
                extra = len(dim_bounds) % (self.m + 1)
                # select the min and max values for each interval
                start_pos = 0
                end_pos = 0
                for i in range(self.m):
                    # distribute extra bounds to the first intervals
                    new_size = group_size
                    if extra > 0:
                        new_size += 1
                        extra -= 1
                    end_pos = start_pos + new_size
                    # set the lower bound and upper bound on the interval
                    intervals[dim, i, LOWER] = dim_bounds[start_pos]
                    intervals[dim, i, UPPER] = dim_bounds[end_pos]
                    start_pos = end_pos
            # if not, disable indexing on this dimension
            else:
                indexed_dimensions[dim] = False

        return intervals, indexed_dimensions

    def parse_hyperparameters(self, hyperparameters: dict) -> None:
        self.hyperparameters = hyperparameters
        params: dict = hyperparameters.get("FACETIndex")

        # Initial Radius
        if params.get("rbv_initial_radius") is None:
            print("No rbv_initial_radius provided, using 0.05")
            self.initial_radius = 0.05
        else:
            self.initial_radius = params.get("rbv_initial_radius")

        # Radius Growth Method
        if params.get("rbv_radius_growth") is None:
            print("No rbv_radius_growth provided, using Linear")
            self.radius_growth = "Linear"
        else:
            self.radius_growth = params.get("rbv_radius_growth")

        # Number of intervals (m)
        if params.get("rbv_num_interval") is None:
            print("No rbv_num_interval provided, using 4")
            self.m = 4
        else:
            self.m = params.get("rbv_num_interval")

        # print messages
        if params.get("facet_verbose") is None:
            self.verbose = False
        else:
            self.verbose = params.get("verbose")


def have_intersection(rect_a: np.ndarray, rect_b: np.ndarray) -> bool:
    '''
    Returns true iff the overlap between the two rectangles is nonzero

    Parameters
    ----------
    rect_a, rect_b: numpy arrays of shape (ndim, 2) representing min/max values along each axis
    '''
    lowers_match = (rect_a[:, LOWER] <= rect_b[:, UPPER]).all()
    uppers_match = (rect_a[:, UPPER] >= rect_b[:, LOWER]).all()
    return lowers_match and uppers_match
