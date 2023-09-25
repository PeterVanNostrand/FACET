# distutils: language=c++

from .extraction_problem import ExtractionProblem, FeatureConditions, SplitPoint
from typing import List, Tuple
# from libcpp.vector cimport vector as cppvector
# from libcpp.pair cimport pair as cpppair
# from libc.math cimport sqrt, fabs, floor, ceil
# from libcpp cimport bool
from typing import List, Tuple

CATEGORY_ACTIVATED: int
CATEGORY_DEACTIVATED: int
CATEGORY_UNSET: int
CATEGORY_UNKNOWN: int
ACTIVATED_MASK: int
BITS_PER_STATE: int


# def partial_distance(problem: ExtractionProblem, feature: int, to_explain: float, counterfactual: float) -> float:
#     pass


# def distance(problem: ExtractionProblem, to_explain: List[float], counterfactual: List[float]) -> float:
#     pass


# def does_meet_split(problem: ExtractionProblem, counterfactual: List[float], split_point: SplitPoint) -> bool:
#     pass


# def update_to_meet_split(problem: ExtractionProblem, counterfactual: List[float], split_point: SplitPoint,  enable_if_last: bool) -> bool:
#     pass


# def update_to_meet(counterfactual: List[float], feature: int, feature_type: int, value: float, meet: bool,  is_mask_set: bool, num_categories: int, enable_if_last: bool, epsilon: float) -> bool:
#     pass


# def is_category_state(category_mask: int, category: int, state: int) -> bool:
#     pass


# def set_category_state(category_mask: int, category: int, state: int) -> int:
#     pass


# def calculate_sorted_rule_distances(problem: ExtractionProblem, observation,
#                                     parsed_rf, dataset_info) -> List[Tuple[int, float]]:
#     pass
