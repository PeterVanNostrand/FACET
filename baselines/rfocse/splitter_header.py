# distutils: language=c++

from dataclasses import dataclass
from .extraction_problem import SplitPoint

# from libcpp.pair cimport pair as cpppair
# from libcpp.vector cimport vector as List
# from libc.math cimport log2, fabs
# from libcpp cimport bool

from typing import List, Tuple


@dataclass
class PartitionProb:
    tree_prob_sum: List[List[float]]
    active_by_tree: List[int]
    prob: List[float]
    num_active: int


@dataclass
class SplitPointScore:
    score: float
    split_point: SplitPoint
    is_ok: bool


# def calculate_split(global_state: ExtractionContext, verbose: bool) -> Tuple[bool, SplitPoint]:
#     pass


# def criterion(label_probs: List[float]) -> float:
#     pass
