# distutils: language=c++

# from libcpp.vector cimport vector as cppvector
# from libcpp.pair cimport pair as cpppair
# from libcpp cimport bool
# from libc.stdio cimport printf
# from libcpp.algorithm cimport lower_bound
# from libc.math cimport sqrt, fabs, floor, ceil


# from .extraction_problem cimport ExtractionProblem, Solution, SplitPoint, FeatureBounds, FeatureConditions,  ConditionSide, ExtractionContext, estimate_probability
# from .math_utils cimport sum_vector, sub_vector, double_argmax
# from .splitter cimport calculate_split, SplitPointScore
# from .observations cimport does_meet_split, update_to_meet_split, CATEGORY_ACTIVATED, CATEGORY_DEACTIVATED, CATEGORY_UNSET, \
#     CATEGORY_UNKNOWN, is_category_state, set_category_state

# from .observations cimport distance as rule_distance
# from .observations cimport partial_distance

from typing import List, Tuple
from .extraction_problem_header import *
from dataclasses import dataclass


@dataclass
class StepState:
    removed_rules: List[int]
    previous_bounds: List[FeatureBounds]


# def activate_rule(global_state: ExtractionContext, rule_id: int) -> bool:
#     pass


# def deactivate_rule(global_state: ExtractionContext, rule_id: int) -> None:
#     pass


# def early_stop(global_state: ExtractionContext) -> int:
#     pass


# def can_stop(global_state: ExtractionContext) -> int:
#     pass


# def is_bucket_active(global_state: ExtractionContext, condition_side: ConditionSide, bucket: int) -> bool:
#     pass


# def soft_pruning(global_state: ExtractionContext, solution: Solution, removed_rules: List[int]) -> None:
#     pass


# def ensure_bounds_consistency(global_state: ExtractionContext, solution: Solution) -> None:
#     pass


# def deactivate_bucket_batch(global_state: ExtractionContext, condition_side: ConditionSide, bucket_from: int, bucket_to: int, removed_rules: List[int]) -> None:
#     pass


# def deactivate_bucket(global_state: ExtractionContext, condition_side: ConditionSide, bucket: int, removed_rules: List[int]) -> None:
#     pass


# def apply_feature_bounds_categorical(global_state: ExtractionContext, split_point: SplitPoint) -> StepState:
#     pass


# def apply_feature_bounds_numerical(global_state: ExtractionContext, split_point: SplitPoint) -> StepState:
#     pass


# def apply_feature_bounds(global_state: ExtractionContext, solution: Solution, SplitPoint & split_point) -> StepState:
#     pass


# def rollback_split(global_state: ExtractionContext, step_state: StepState) -> None:
#     pass


# def get_set_trees_ids(global_state: ExtractionContext) -> List[int]:
#     pass


# def calculate_current_distance(global_state: ExtractionContext) -> float:
#     pass


# def prune_rules(global_state: ExtractionContext, solution: Solution) -> bool:
#     pass


# def extract_counterfactual_impl(global_state: ExtractionContext, solution: Solution) -> None:
#     pass
