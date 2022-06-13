# handle circular imports that result from typehinting
from __future__ import annotations

import numpy as np
from bitarray import bitarray
from bitarray.util import zeros as bitzeros

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # circular import avoidance
    from explainers.facet_index import FACETIndex

LOWER = 0
UPPER = 1


class BitVectorIndex():
    '''
    A method for performing high dimensional indexing of hyper-rectangles using a set of precomputed redundant bit vectors. Designed to efficiently find the nearest hyperrectangle to a point subject to an optional set of constraints along each axis.

    Based on "Indexing High Dimensional Rectangles for Fast Multimedia Identification" by Jonathan Goldstein, John Platt, Christopher Burges. 2003 Microsoft Research tecnical report
    '''

    def __init__(self, rects: list[np.ndarray], explainer: FACETIndex, m: int = 16, initial_radius=0.05):
        '''
        Parameters
        ----------
        rects: the list of hyperrectangle records to index, all records should be of the same class
        m: the nuymber of intervals to create along each dimension
        '''
        self.explainer = explainer
        self.initial_radius = initial_radius
        self.m = m
        # combine the list of hyperrectangles into one array of shape (nrects, ndim, 2)
        self.rects: np.ndarray = np.stack(rects, axis=0)
        self.nrects = self.rects.shape[0]
        self.ndimensions = self.rects.shape[1]

        # select the partition values for each interval
        self.intervals, self.indexed_dimensions = self.generate_intervals(self.rects)
        print("N Indexed Dimensions:", sum(self.indexed_dimensions))
        self.rbv = self.build_bit_vectors(self.rects)

    def point_query(self, instance: np.ndarray):
        '''
        Uses the bit vector index to find hyper-rectangles which are close to the given point

        Parameters
        ----------
        instance: a numpy array of shape (ndim,) to search around

        Returns
        -------
        matching_rects: a numpy array of shape(nrecords, ndim, 2) of the nearby hyper-rectangles
        '''
        # create a hyper-sphere around the point and indentify which intervals it covers. We do this by creating a hyper-sphere with the initial radius, converting it to a hyper-rectangle, and searching for records in that rect
        search_radius = self.initial_radius
        closest_rect = None
        solution_found = False
        n_rects_checked = 0
        while not solution_found:  # if at least one bit is set, a HR record was found in our search region
            query_rect = np.zeros(shape=(self.ndimensions, 2))
            query_rect[:, LOWER] = (instance - search_radius)
            query_rect[:, UPPER] = (instance + search_radius)
            matching_bits = self.rect_query(query_rect)
            if matching_bits.any():
                n_rects_checked += matching_bits.count()
                # expand the packed bitarry to an array of booleans
                matching_slice = np.array(matching_bits.tolist(), dtype=bool)
                # get the matching rectangles
                matching_rects = self.rects[matching_slice]
                # search the matching rects for the nearest rectangle within the search radius ts possible that the matching set is non-empty due to a rectangle in an unindexed dimension that is further than the search radius, which is not guaranteed to be the nearest to the point
                closest_rect = None
                min_dist = np.inf
                for rect in matching_rects:
                    test_instance = self.explainer.fit_to_rectangle(instance, rect)
                    dist = self.explainer.distance_fn(instance, test_instance)
                    # valid solutions must fall within the search radius
                    if dist < search_radius and dist < min_dist:
                        solution_found = True
                        min_dist = dist
                        closest_rect = rect
            search_radius += self.initial_radius
        print(matching_rects.shape[0], n_rects_checked)
        return closest_rect

    def rect_query(self, query_rect: np.ndarray):
        '''
        Finds the set of all record hyper-rectangles which overlap the region defined in the query rectangle

        Parameters
        ----------
        query_rect: a numpy array of shape (ndim, 2) representing the upper/lower bound along each axis to search in

        Returns
        -------
        matching_bits: a bitarray of length nrects with each bit set to one iff the corresponding hyper-rectangle fall in the query region
        '''
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

    def build_bit_vectors(self, rects: np.ndarray):
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
            # if not split the space evenly between the intervals
            # else:
            #     # remove +/- infinite values
            #     idx_neg_inf = np.argwhere(dim_bounds != np.inf)
            #     if idx_neg_inf.shape[0] > 0:
            #         dim_bounds = dim_bounds[idx_neg_inf].squeeze()
            #     idx_pos_inf = np.argwhere(dim_bounds != -np.inf)
            #     if idx_pos_inf.shape[0] > 0:
            #         dim_bounds = dim_bounds[idx_pos_inf].squeeze()

            #     # there are at lest two bounds use them to set the inteval width
            #     if dim_bounds.size >= 2:
            #         min_val = np.min(dim_bounds)
            #         data_range = np.max(dim_bounds) - min_val
            #     # otherwise use the range zero to one
            #     else:
            #         min_val = 0
            #         data_range = 1 - min_val
            #     interval_width = data_range / (self.m - 1)
            #     # assign lower and upper bounds on intervals
            #     for i in range(self.m):
            #         if i == 0:
            #             intervals[dim, i, LOWER] = -np.inf
            #         else:
            #             intervals[dim, i, LOWER] = min_val + (i-1) * interval_width

            #         if i == (self.m-1):
            #             intervals[dim, i, UPPER] = np.inf
            #         else:
            #             intervals[dim, i, UPPER] = min_val + i * interval_width
        return intervals, indexed_dimensions
