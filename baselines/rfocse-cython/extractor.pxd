# distutils: language=c++

from libcpp.vector cimport vector as cppvector
from libcpp.pair cimport pair as cpppair
from libcpp cimport bool
from libc.stdio cimport printf
from libcpp.algorithm cimport lower_bound
from libc.math cimport sqrt, fabs, floor, ceil


from .extraction_problem cimport ExtractionProblem, Solution, SplitPoint, FeatureBounds, FeatureConditions,  ConditionSide, ExtractionContext, estimate_probability
from .math_utils cimport sum_vector, sub_vector, double_argmax
from .splitter cimport calculate_split, SplitPointScore
from .observations cimport does_meet_split, update_to_meet_split, CATEGORY_ACTIVATED, CATEGORY_DEACTIVATED, CATEGORY_UNSET, \
    CATEGORY_UNKNOWN, is_category_state, set_category_state

from .observations cimport distance as rule_distance
from .observations cimport partial_distance



cdef struct StepState:
    cppvector[int] removed_rules
    cppvector[FeatureBounds] previous_bounds



cdef bool activate_rule(ExtractionContext  & global_state, int rule_id) 

cdef void deactivate_rule(ExtractionContext  & global_state, int rule_id) 

cdef int early_stop(ExtractionContext  & global_state)

cdef int can_stop(ExtractionContext  & global_state)

cdef bool is_bucket_active(ExtractionContext & global_state, ConditionSide & condition_side, int bucket)

cdef void soft_pruning(ExtractionContext & global_state, Solution & solution, cppvector[int] & removed_rules)

cdef void ensure_bounds_consistency(ExtractionContext & global_state, Solution & solution)

cdef void deactivate_bucket_batch(ExtractionContext & global_state, ConditionSide & condition_side, int bucket_from, int bucket_to, cppvector[int] & removed_rules)

cdef void deactivate_bucket(ExtractionContext & global_state, ConditionSide & condition_side, int bucket, cppvector[int] & removed_rules)

cdef StepState apply_feature_bounds_categorical(ExtractionContext & global_state, SplitPoint & split_point)

cdef StepState apply_feature_bounds_numerical(ExtractionContext & global_state, SplitPoint & split_point)

cdef StepState apply_feature_bounds(ExtractionContext & global_state, Solution & solution, SplitPoint & split_point)

cdef void rollback_split(ExtractionContext  & global_state, StepState & step_state)

cdef cppvector[int] get_set_trees_ids(ExtractionContext & global_state)

cdef double calculate_current_distance(ExtractionContext & global_state)

cdef bool prune_rules(ExtractionContext & global_state, Solution & solution)

cdef void extract_counterfactual_impl(ExtractionContext & global_state, Solution & solution)
