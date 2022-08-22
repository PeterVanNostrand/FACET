# distutils: language=c++

from .extraction_problem cimport ExtractionProblem, FeatureConditions, SplitPoint
from libcpp.vector cimport vector as cppvector
from libcpp.pair cimport pair as cpppair
from libc.math cimport sqrt, fabs, floor, ceil
from libcpp cimport bool
from .debug cimport ExtractionProblem_tostr, DatasetInfo_tostr, LOG_LEVEL, instance_num

cpdef long long CATEGORY_ACTIVATED
cpdef long long CATEGORY_DEACTIVATED
cpdef long long CATEGORY_UNSET
cpdef long long CATEGORY_UNKNOWN
cpdef long long ACTIVATED_MASK
cpdef long long BITS_PER_STATE


cdef double partial_distance(ExtractionProblem & problem, int feature, double to_explain, double counterfactual)

cdef double distance(ExtractionProblem & problem, cppvector[double] & to_explain, cppvector[double] & counterfactual)

cdef bool does_meet_split(ExtractionProblem & problem, cppvector[double] & counterfactual, SplitPoint & split_point)

cdef bool update_to_meet_split(ExtractionProblem & problem, cppvector[double] & counterfactual, SplitPoint & split_point, bool enable_if_last=*)

cdef bool update_to_meet(cppvector[double] & counterfactual, int feature, int feature_type, double value, bool meet, bool is_mask_set=*, int num_categories=*, bool enable_if_last=*, double epsilon=*)

cdef bool is_category_state(long long category_mask, int category, int state) 

cdef long long set_category_state(long long category_mask, int category, int state) 

cdef cppvector[cpppair[int, double]] calculate_sorted_rule_distances(ExtractionProblem & problem, observation,
                                                                     parsed_rf, dataset_info)