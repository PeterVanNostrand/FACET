from typing import List

import numpy as np


def sum_vector(labels: List[float], into: List[float]) -> None:
    for i in range(len(labels)):
        into[i] = into[i] + labels[i]


def sub_vector(labels: List[float], into: List[float]) -> None:
    for i in range(len(labels)):
        into[i] = into[i] - labels[i]


def cmp(v1: float, v2: float, is_lte: bool) -> bool:
    if is_lte:
        return v1 <= v2
    else:
        return v1 > v2


def float_argmax(v: List[float]) -> int:
    max_pos: int = -1
    max_value: float = -np.inf

    for i in range(len(v)):
        if max_pos == -1 or v[i] > max_value:
            max_pos = i
            max_value = v[i]

    return max_pos


def is_number(a):
    try:
        int(repr(a))
    except:
        return False

    return True
