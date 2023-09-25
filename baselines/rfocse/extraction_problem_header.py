# distutils: language=c++

# from libcpp.vector cimport vector as cppvector
# from libcpp.pair cimport pair as cpppair
# from libcpp cimport bool
# from libc.math cimport sqrt, fabs, floor, ceil

# from .math_utils cimport sum_vector, double_argmax


from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FeatureBounds:
    numerical_bounds: List[int]
    categorical_mask: int
    is_mask_set: bool


@dataclass
class SplitPoint:
    value: float
    feature: int
    meet: bool
    bounds_meet: FeatureBounds
    bounds_not_meet: FeatureBounds


@dataclass
class ValueIdsRanges:
    value: float
    ids_starts_at: int
    ids_ends_at: int
    size: int


@dataclass
class ConditionSide:
    ids_values: List[ValueIdsRanges]
    ids: List[int]


@dataclass
class FeatureConditions:
    feature_type: int

    side_1: ConditionSide
    side_2: ConditionSide

    feat_range: float


@dataclass
class Solution:
    conditions: List[SplitPoint]
    set_rules: List[int]
    instance: List
    distance: float
    found: bool
    num_evaluations: int
    num_discovered_non_foil: int
    estimated_class: int


@dataclass
class ExtractionProblem:
    feature_conditions: List[FeatureConditions]
    rule_probabilities: List[List[float]]
    rule_tree: List[int]

    n_rules: int
    n_features: int
    n_labels: int
    n_trees: int

    search_closest: bool

    log_every: int

    max_iterations: int


@dataclass
class ExtractionContext:
    problem: ExtractionProblem
    active_rules: List[bool]
    feature_bounds: List[FeatureBounds]
    feature_conditions_view: List[FeatureConditions]

    tree_prob_sum: List[List[float]]
    active_by_tree: List[int]
    splits: List[SplitPoint]

    num_active_rules: int

    current_obs: List[float]

    to_explain: List[float]
    real_class: int

    sorted_rule_distances: List[Tuple[int, float]]
    current_rule_pruning: int
    rule_blacklist: List[bool]

    max_distance: float

    sorted_rule_end_idx: int

    debug: bool


# def estimate_probability(global_state: ExtractionContext)-> List[float]:
#     pass
