import math
import numpy as np
from operator import itemgetter
from typing import List, Tuple
from .extractor_header import ExtractionProblem, FeatureConditions, SplitPoint
from .debug import LOG_LEVEL, ExtractionProblem_tostr, DatasetInfo_tostr, instance_num

CATEGORY_ACTIVATED = 2
CATEGORY_DEACTIVATED = 1
CATEGORY_UNKNOWN = 3
CATEGORY_UNSET = 0
ACTIVATED_MASK = 0xAAAAAAAAAAAAAAAA


def partial_distance(problem: ExtractionProblem, feature: int, to_explain: float, counterfactual: float) -> float:
    num_features: int = problem.n_features
    feature_conditions: FeatureConditions = problem.feature_conditions[feature]
    feature_type: int = feature_conditions.feature_type
    feat_distance: float

    if feature_type == 4:
        if (ACTIVATED_MASK & (to_explain) & (counterfactual)) == 0:
            feat_distance = 1
    if feature_type == 5:
        return 0.0 if to_explain == counterfactual else 1.0
    else:
        feat_distance = abs(to_explain - counterfactual) / feature_conditions.feat_range
        feat_distance = feat_distance

    return feat_distance / num_features


def distance(problem: ExtractionProblem, to_explain: List[float], counterfactual: List[float]) -> float:
    feature_conditions: FeatureConditions
    distance: float = 0
    num_features: int = problem.n_features
    feature: int
    feature_type: int
    feat_distance: float
    for feature in range(num_features):
        distance += partial_distance(problem, feature, to_explain[feature], counterfactual[feature])
    return distance


def does_meet_split(problem: ExtractionProblem, counterfactual: List[float], split_point: SplitPoint) -> bool:
    feature: int = split_point.feature
    feature_conditions: FeatureConditions = problem.feature_conditions[feature]
    feature_type: int = feature_conditions.feature_type
    categorical_mask: int

    category: int
    state: int

    if feature_type == 4:
        categorical_mask = counterfactual[split_point.feature]
        category = split_point.value

        if split_point.meet:
            return is_category_state(categorical_mask, category, CATEGORY_ACTIVATED)
        else:
            return not is_category_state(categorical_mask, category, CATEGORY_ACTIVATED)
    else:
        return split_point.meet == (counterfactual[feature] <= split_point.value)


def update_to_meet_split(problem: ExtractionProblem, counterfactual: List[float],
                         split_point: SplitPoint, enable_if_last: bool = False) -> bool:
    feature: int = split_point.feature
    feature_conditions: FeatureConditions = problem.feature_conditions[feature]
    feature_type: int = feature_conditions.feature_type
    is_mask_set: bool = split_point.bounds_meet.is_mask_set if split_point.meet else split_point.bounds_not_meet.is_mask_set
    return update_to_meet(counterfactual, feature, feature_type, split_point.value, split_point.meet,
                          is_mask_set, problem.n_features, enable_if_last)


def update_to_meet(counterfactual: List[float], feature: int, feature_type: int,
                   value: float, meet: bool, is_mask_set: bool = False, num_categories: int = -1,
                   enable_if_last: bool = False, epsilon: float = 0.0) -> bool:
    counter_start = counterfactual.copy()
    offset: float = 0
    position: int
    updated: bool = False
    category_mask: int

    if feature_type == 4 and value >= 20:
        print("Cannot work with categorical features with more than 20 levels")
        exit(0)

    if feature_type == 1:
        if (meet and value < counterfactual[feature]) or (not meet and value >= counterfactual[feature]):
            if feature_type == 1 and not meet:
                offset = epsilon

            counterfactual[feature] = value + offset
            updated = True
    elif feature_type == 2 or feature_type == 3:  # ? MODIFIED FROM 'AND' TO 'OR' DUE TO CYTHON BUG
        if (meet and value < counterfactual[feature]) or (not meet and value >= counterfactual[feature]):
            counterfactual[feature] = math.floor(value) if meet else math.floor(value + 1)
            updated = True
    elif feature_type == 4:
        position = value
        category_mask = counterfactual[feature]
        if not meet:
            if enable_if_last and is_mask_set and num_categories != -1:
                for position in range(num_categories):
                    if is_category_state(category_mask, position, CATEGORY_UNSET):
                        counterfactual[feature] = set_category_state(0, position, CATEGORY_ACTIVATED)
                        updated = True
                        break
            else:
                counterfactual[feature] = set_category_state(category_mask, position,
                                                             CATEGORY_DEACTIVATED)
                updated = True
        else:
            counterfactual[feature] = set_category_state(0, position, CATEGORY_ACTIVATED)
            updated = True

    return updated


def is_category_state(category_mask: int, category: int, state: int) -> bool:
    if state == CATEGORY_UNSET:
        return (category_mask & (3 << 2 * category)) == 0

    activated: bool = (category_mask & (CATEGORY_ACTIVATED << 2 * category)) > 0
    deactivated: bool = (category_mask & (CATEGORY_DEACTIVATED << 2 * category)) > 0

    if state == CATEGORY_UNKNOWN:
        return activated and deactivated
    elif state == CATEGORY_DEACTIVATED:
        return deactivated and not activated
    else:
        return activated and not deactivated


def set_category_state(category_mask: int, category: int, state: int) -> int:
    clean_mask: int = ~(3 << 2 * category)
    return (category_mask & clean_mask) | (state << 2 * category)


def adapt_observation_representation(observation, dataset_info, include_all_categories=False):
    res = np.zeros(len(dataset_info.dataset_description), dtype=np.float)
    for feature, feature_info in dataset_info.dataset_description.items():
        if feature_info['type'] in (1, 2, 3):
            res[feature] = observation[feature_info['original_position']]
        else:
            if include_all_categories:
                for i in range(len(feature_info['categories_original_position'])):
                    res[feature] = float(int(res[feature]) | 2 << (2 * i))
            else:
                found = False
                for i, pos in enumerate(feature_info['categories_original_position']):
                    if observation[pos] > 0:
                        found = True
                        break

                if not found:
                    raise ValueError("One-hot encoded variable does not have any category "
                                     "active ({})".format(str(feature_info['categories_original_position'])))

                res[feature] = float(2 << (2 * i))
    return res


def calculate_sorted_rule_distances(problem: ExtractionProblem, observation,
                                    parsed_rf, dataset_info) -> List[Tuple[int, float]]:

    if LOG_LEVEL == "DEBUG":
        frule = open("logs/py_log_{}.txt".format(instance_num), "a")
        frule.write("######################### CALC SORTED RULE DISTANCE #########################\n")
        frule.write(ExtractionProblem_tostr(problem))
        frule.write("observation: " + str(observation) + "\n")
        frule.write("dataset_info:\n")
        frule.write(DatasetInfo_tostr(dataset_info))
        frule.write("parsed_rf:\n")
        frule.write(str(parsed_rf) + "\n")
        frule.close()

    rule_distances = []

    to_explain: List[float] = adapt_observation_representation(observation, dataset_info,
                                                               include_all_categories=False)

    if LOG_LEVEL == "DEBUG":
        frule = open("logs/py_log_{}.txt".format(instance_num), "a")
        out_str = "to_explain: ["
        for i in range(len(to_explain)):
            out_str += str(to_explain[i])
            if i != len(to_explain)-1:
                out_str += ", "
        out_str += "]\n"
        frule.write(out_str)
        frule.close()

    # current_obs: List[float]
    # rule_dist: float
    for r_id, (t_id, rule) in enumerate(parsed_rf.rules):
        current_obs: List[float] = adapt_observation_representation(
            observation, dataset_info, include_all_categories=True)

        for feature_orig, threshold, is_leq in rule.conditions:
            feature = dataset_info.inverse_dataset_description[feature_orig]['current_position']
            attr_info = dataset_info.dataset_description[feature]
            feature_type = attr_info['type']

            if attr_info['type'] == 4:
                idx = attr_info['categories_original_position'].index(feature_orig)
                value = float(idx)
                meet_cond = not is_leq
            else:
                value = threshold
                meet_cond = is_leq

            previous_value = current_obs[feature]
            update_to_meet(current_obs, feature, feature_type, value, meet_cond)

        if LOG_LEVEL == "DEBUG":
            frule = open("logs/py_log_{}.txt".format(instance_num), "a")
            out_str = "obs: ["
            for i in range(len(current_obs)):
                out_str += str(current_obs[i])
                if i != len(current_obs)-1:
                    out_str += ", "
            out_str += "]\n"
            frule.write(out_str)
            frule.close()

        rule_dist: float = distance(problem, to_explain, current_obs)
        rule_distances.append((r_id, rule_dist))

    rule_distances = sorted(rule_distances, key=itemgetter(1))
    rule_distances_c: List[Tuple[int, float]] = []

    all_zero = True
    for r_id, r_distance in rule_distances:
        rule_distances_c.append((r_id, r_distance))
        if LOG_LEVEL == "DEBUG":
            frule = open("logs/py_log_{}.txt".format(instance_num), "a")
            frule.write(("rid: {}, rdist: {}\n".format(r_id, r_distance)))
            frule.close()
            if r_distance > 0:
                all_zero = False

    if LOG_LEVEL == "DEBUG":
        if all_zero:
            print("ALL ZERO CASE")
            print(str(to_explain))
            print(str(current_obs))

    return rule_distances_c
