# distutils: language=c++

from libcpp.vector cimport vector as cppvector
from libcpp.pair cimport pair as cpppair
from libcpp cimport bool
from libc.math cimport sqrt, fabs, floor, ceil

from .math_utils cimport sum_vector, double_argmax



cdef struct FeatureBounds:
    cpppair[int, int] numerical_bounds
    long long categorical_mask
    bool is_mask_set

cdef struct SplitPoint:
    double value
    int feature
    bool meet
    FeatureBounds bounds_meet
    FeatureBounds bounds_not_meet

cdef struct ValueIdsRanges:
    double value
    int ids_starts_at
    int ids_ends_at
    int size

cdef struct ConditionSide:
    cppvector[ValueIdsRanges] ids_values
    cppvector[int] ids

cdef struct FeatureConditions:
    int feature_type

    ConditionSide side_1
    ConditionSide side_2

    double feat_range

cdef struct Solution:
    cppvector[SplitPoint] conditions
    cppvector[int] set_rules
    cppvector[double] instance
    double distance
    bool found
    long num_evaluations
    long num_discovered_non_foil
    int estimated_class


cdef struct ExtractionProblem:
    cppvector[FeatureConditions] feature_conditions
    cppvector[cppvector[double]] rule_probabilities
    cppvector[int] rule_tree


    int n_rules
    int n_features
    int n_labels
    int n_trees

    bool search_closest

    int log_every

    int max_iterations

cdef struct ExtractionContext:
    ExtractionProblem problem
    cppvector[bool] active_rules
    cppvector[FeatureBounds] feature_bounds
    cppvector[FeatureConditions] feature_conditions_view

    cppvector[cppvector[double]] tree_prob_sum
    cppvector[int] active_by_tree
    cppvector[SplitPoint] splits

    int num_active_rules

    cppvector[double] current_obs

    cppvector[double] to_explain
    int real_class

    cppvector[cpppair[int, double]] sorted_rule_distances
    int current_rule_pruning
    cppvector[bool] rule_blacklist

    double max_distance

    int sorted_rule_end_idx

    bool debug

cdef cppvector[double] estimate_probability(ExtractionContext & global_state) 

cdef ConditionSide make_condition_side(condition_side, int feature, int feature_type, bool is_side_1)

cdef ExtractionProblem make_problem(parsed_rf, dataset_description,
                                    bool search_closest_c, int log_every_c, int max_iterations_c)

cdef ExtractionContext create_extraction_state(ExtractionProblem & problem, parsed_rf, factual_class, observation, dataset_info, max_distance)
