# distutils: language=c++

from .extraction_problem cimport FeatureConditions, FeatureBounds, SplitPoint, ConditionSide, \
    ExtractionContext, estimate_probability
from .observations cimport CATEGORY_ACTIVATED, CATEGORY_DEACTIVATED, CATEGORY_UNSET, CATEGORY_UNKNOWN,\
    is_category_state, set_category_state


from libcpp.pair cimport pair as cpppair
from libcpp.vector cimport vector as cppvector
from libc.math cimport log2, fabs
from libcpp cimport bool




cdef struct PartitionProb:
    cppvector[cppvector[double]] tree_prob_sum
    cppvector[int] active_by_tree
    cppvector[double] prob
    int num_active


cdef struct SplitPointScore:
    double score
    SplitPoint split_point
    bool is_ok

cdef cpppair[bool, SplitPoint] calculate_split(ExtractionContext & global_state, bool verbose=*)

cdef double criterion(cppvector[double] & label_probs)
