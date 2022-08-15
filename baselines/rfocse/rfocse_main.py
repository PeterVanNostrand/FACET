from typing import List, Tuple
import time
import random
import math
import numpy as np
from itertools import product
from collections import Counter
from .extraction_problem_header import ExtractionProblem
from .observations import adapt_observation_representation
from .observations import distance
from .converter import convert_rf_format, export
from .extractor import extract_counterfactual_impl
from .extraction_problem import make_problem, create_extraction_state, ExtractionProblem, ExtractionContext, Solution, SplitPoint

# from libcpp.vector cimport vector as cppvector
# from libcpp.pair cimport pair as cpppair


def get_closest_counterfactual(parsed_rf, X, sample, dataset_info, problem: ExtractionProblem) -> Tuple[int, float]:
    rf = parsed_rf.sklearn_model
    ys = rf.predict(X)
    sample_class = rf.predict([sample])[0]
    counterfactuals = [i for i, pred in enumerate(ys) if pred != sample_class]

    sample_c: List[float] = adapt_observation_representation(sample, dataset_info)

    best_idx: int = counterfactuals[0]
    counterfactual_c: List[float] = adapt_observation_representation(X[best_idx], dataset_info)
    best_distance: float = distance(problem, sample_c, counterfactual_c)
    current_distance: float

    for x_idx in counterfactuals[1:]:
        counterfactual_c = adapt_observation_representation(X[x_idx], dataset_info)
        current_distance = distance(problem, sample_c, counterfactual_c)

        if current_distance < best_distance:
            best_distance = current_distance
            best_idx = x_idx

    return (best_idx, best_distance)


def counterfactual_set_from_obs(parsed_rf, obs, dataset_info, foil_class, filter_rules=True, filter_label_type='prob'):
    leaves = parsed_rf.sklearn_model.apply([obs])[0]
    rule_ids = [parsed_rf.sklearn_rule_mapping[t_id][l_id] for t_id, l_id in enumerate(leaves)]

    if filter_rules:

        def get_label_imp(label):
            if filter_label_type == 'prob':
                return (label.reshape(1, -1) / label.sum(-1)).reshape(-1)[foil_class]
            elif filter_label_type == 'count':
                return label.reshape(-1)[foil_class]
            else:
                raise ValueError("Unknown label agg type {tt}".format(tt=filter_label_type))

        rule_prob = [(rule.label.reshape(1, -1) / rule.label.sum(-1)).reshape(-1)[foil_class]
                     for rule, rid in [(parsed_rf.rules[rid][1], rid) for rid in rule_ids]]
        rule_imp = [(get_label_imp(rule.label), rule.area, rid)
                    for rule, rid in [(parsed_rf.rules[rid][1], rid) for rid in rule_ids]]
        rule_imp = sorted(rule_imp, reverse=True)

        i = 0
        acc_prob = 0
        cutoff_prob = len(leaves) / 2
        while i < len(rule_imp) and acc_prob < cutoff_prob:
            acc_prob += rule_prob[i]
            i += 1

        rule_ids = [rid for _, _, rid in rule_imp[:i]]

    return build_counterfactual_set(parsed_rf, list(rule_ids), dataset_info)


def batch_extraction(sklearn_rf, dataset_info, X, max_distance=-1,
                     search_closest=True, log_every=50_000, max_iterations=-1, export_rules_file=None,
                     dataset=None, epsilon=0.0005):
    '''
    Parameters
    ----------
    sklearn_rf
    dataset_info
    X
    max_distance
    search_closest
    log_every
    max_iterations
    export_rules_file
    dataset
    epsilon
    '''
    parsed_rf = convert_rf_format(sklearn_rf, dataset_info)

    if export_rules_file is not None:
        export(parsed_rf.rules, export_rules_file, dataset_info)

    problem: ExtractionProblem = make_problem(parsed_rf, dataset_info, search_closest, log_every,
                                              max_iterations)

    for idx, observation in enumerate(X):

        start = time.time()
        res = extract_single_counterfactual_set(problem, sklearn_rf, parsed_rf, dataset_info, observation, dataset)
        end = time.time()
        res['_idx'] = idx
        res['extraction_time'] = end - start

        yield res


def extract_single_counterfactual_set(problem: ExtractionProblem, sklearn_rf, parsed_rf, dataset_info, to_explain, dataset):
    closest_mo: Tuple[int, float]

    if dataset is not None:
        closest_mo = get_closest_counterfactual(parsed_rf, dataset, to_explain, dataset_info, problem)

    factual_class = int(sklearn_rf.predict([to_explain])[0])

    global_state: ExtractionContext = create_extraction_state(problem, parsed_rf, factual_class, to_explain,
                                                              dataset_info, closest_mo[1])

    solution: Solution = Solution(
        [],  # cppvector[SplitPoint] conditions
        [],  # cppvector[int] set_rules
        to_explain,  # cppvector[double] instance
        -1,  # double distance
        False,  # bool found
        0,  # long num_evaluations
        0,  # long num_discovered_non_foil
        -1  # int estimated_class
    )

    solution.found = False
    extract_counterfactual_impl(global_state, solution)

    res = {
        'found': solution.found,
        'num_iterations': solution.num_evaluations,
        'num_foil': solution.num_discovered_non_foil,
        '_observation': to_explain,
        '_idx': 0,
        'factual_class': factual_class
    }

    if solution.found:

        split_point_info = []

        for split_point in solution.conditions:
            split_point_info.append((split_point.feature, split_point.value, split_point.meet))

        res['explanation'] = build_counterfactual_set(parsed_rf, list(solution.set_rules), dataset_info, obs=solution.instance,
                                                      split_point_info=split_point_info)
        res['explanation_compact'] = counterfactual_set_from_obs(parsed_rf,
                                                                 res['explanation'].sample_counterfactual(to_explain),
                                                                 dataset_info, (factual_class + 1) % 2)

        res['explanation_compact_count'] = counterfactual_set_from_obs(parsed_rf,
                                                                       res['explanation'].sample_counterfactual(
                                                                           to_explain),
                                                                       dataset_info, (factual_class + 1) % 2,
                                                                       filter_label_type='count')

        res['feature_bounds_splits'] = rule_set_from_split_points(split_point_info, dataset_info)
        res['splits'] = split_point_info
        res['remaining_rules'] = list(solution.set_rules)
        res['set_rules'] = res['explanation'].rule_ids
        res['foil_estimated_class'] = solution.estimated_class,
        res['distance'] = solution.distance
    elif dataset is not None:
        res['explanation'] = counterfactual_set_from_obs(
            parsed_rf, dataset[closest_mo[0]], dataset_info, (factual_class + 1) % 2)
        res['explanation_compact'] = res['explanation']
        res['explanation_compact_count'] = counterfactual_set_from_obs(parsed_rf,
                                                                       dataset[closest_mo[0]],
                                                                       dataset_info, (factual_class + 1) % 2,
                                                                       filter_label_type='count')
        res['feature_bounds_splits'] = res['explanation_compact'].feat_conditions
        res['remaining_rules'] = res['explanation'].rule_ids
        res['set_rules'] = res['explanation'].rule_ids
        res['foil_estimated_class'] = sklearn_rf.predict([dataset[closest_mo[0]]])[0]
        res['distance'] = closest_mo[1]
        res['found'] = True
    return res


def extract_counterfactual(sklearn_rf, dataset_info, to_explain, max_distance=-1,
                           search_closest=True, log_every=50_000, max_iterations=-1, export_rules_file=None,
                           dataset=None):
    parsed_rf = convert_rf_format(sklearn_rf, dataset_info)

    if export_rules_file is not None:
        export(parsed_rf.rules, export_rules_file, dataset_info)

    problem: ExtractionProblem = make_problem(parsed_rf, dataset_info, search_closest, log_every,
                                              max_iterations)

    return extract_single_counterfactual_set(problem, sklearn_rf, parsed_rf, dataset_info, to_explain, dataset)


def meets_rule(rule, obs, split_point_info):
    for condition in rule.conditions:
        strict_gt = (condition.feature, condition.threshold, False) not in split_point_info
        if condition.is_leq and condition.threshold < obs[condition.feature]:
            return False
        elif not condition.is_leq and strict_gt and condition.threshold >= obs[condition.feature]:
            return False
        elif not condition.is_leq and not strict_gt and condition.threshold > obs[condition.feature]:
            return False

    return True


def rule_set_from_split_points(split_points, dataset_info):
    rule_set = create_empty_rule_set(dataset_info)

    for feature, threshold, is_leq in split_points:
        add_condition_ruleset(rule_set, feature, threshold, is_leq, dataset_info)

    return rule_set


def create_empty_rule_set(dataset_info):
    return {
        attr: {
            'type': attr_info['type'],
            'values': [None, None] if attr_info['type'] in (1, 2) else list(
                attr_info.get('categories_original_position',
                              [0, 1]))
        }
        for attr, attr_info in dataset_info.dataset_description.items()
    }


def add_condition_ruleset(rule_set, feature, threshold, is_leq, dataset_info):
    inv_attr = dataset_info.inverse_dataset_description[feature]['current_position']
    attr_info = dataset_info.dataset_description[inv_attr]
    feature_ruleset = rule_set[inv_attr]

    if attr_info['type'] in (1, 2):
        if is_leq and (feature_ruleset['values'][1] is None or feature_ruleset['values'][1] > threshold):
            feature_ruleset['values'][1] = threshold
        elif not is_leq and (feature_ruleset['values'][0] is None or feature_ruleset['values'][0] < threshold):
            feature_ruleset['values'][0] = threshold
    elif attr_info['type'] == 3:
        if is_leq:
            feature_ruleset['values'] = [0]
        else:
            feature_ruleset['values'] = [1]
    elif attr_info['type'] == 4:
        if is_leq and feature in feature_ruleset['values']:
            feature_ruleset['values'].remove(feature)
        elif not is_leq:
            feature_ruleset['values'] = [feature]


def build_counterfactual_set(parsed_rf, rule_ids, dataset_info, obs=None, split_point_info=None):
    rule_set = create_empty_rule_set(dataset_info)

    set_trees = Counter([parsed_rf.rules[rule_id][0] for rule_id in rule_ids])

    set_rules = []
    for rule_id in rule_ids:
        t_id, rule = parsed_rf.rules[rule_id]

        if set_trees[t_id] > 1:
            if obs is None or split_point_info is None:
                raise ValueError(
                    "Obs and split_point_info should be provided when there is more than one observation in a tree available")

            if not meets_rule(rule, obs, split_point_info):
                continue

        set_rules.append(rule_id)

        for feature, threshold, is_leq in rule.conditions:
            add_condition_ruleset(rule_set, feature, threshold, is_leq, dataset_info)

    if len(set_rules) != len(set_trees):
        raise ValueError("There is a tree which do not meet the observation {set_rules}, {all_rules}, {obs}, {ssinfo}".format(
            set_rules=set_rules,
            all_rules=rule_ids,
            obs=obs,
            ssinfo=split_point_info
        ))
    return CounterfactualSetExplanation(rule_set, dataset_info, set_rules)


class CounterfactualSetExplanation:

    def __init__(self, feat_conditions, dataset_info, rule_ids):
        self.feat_conditions = feat_conditions
        self.dataset_info = dataset_info
        self.rule_ids = rule_ids

    def sample_counterfactual(self, sample, epsilon=0.001, closest=True):
        obs = np.zeros(len(sample))

        for attr, attr_info in self.feat_conditions.items():
            if attr_info['type'] == 1:
                inv_attr = self.dataset_info.dataset_description[attr]['original_position']
                gt, leq = attr_info['values']

                if gt is not None and (sample[inv_attr] < gt or abs(sample[inv_attr] - gt) < epsilon):
                    obs[inv_attr] = gt + epsilon
                elif leq is not None and (sample[inv_attr] > leq or abs(sample[inv_attr] - leq) < epsilon):
                    obs[inv_attr] = leq - epsilon
                else:
                    obs[inv_attr] = sample[inv_attr]

            elif attr_info['type'] == 2:
                inv_attr = self.dataset_info.dataset_description[attr]['original_position']
                gt, leq = attr_info['values']

                if gt is not None and (sample[inv_attr] < gt or abs(sample[inv_attr] - gt) < epsilon):
                    obs[inv_attr] = math.floor(gt + 1)
                elif leq is not None and (sample[inv_attr] > leq or abs(sample[inv_attr] - leq) < epsilon):
                    obs[inv_attr] = math.floor(leq)
                else:
                    obs[inv_attr] = sample[inv_attr]

            elif attr_info['type'] == 3:
                inv_attr = self.dataset_info.dataset_description[attr]['original_position']
                current_value = int(sample[inv_attr])

                if current_value not in attr_info['values']:
                    obs[inv_attr] = attr_info['values'][0]
                else:
                    obs[inv_attr] = current_value
            else:
                current_category = next(i for i in
                                        self.dataset_info.dataset_description[attr]['categories_original_position']
                                        if sample[i] > 0)
                valid_attrs = attr_info['values']

                if current_category in valid_attrs:
                    obs[current_category] = 1
                else:
                    obs[random.sample(valid_attrs, 1)[0]] = 1

        return obs

    def closest_counterfactuals(self, sample, epsilon=0.001, closest=True):
        obs_def = [None] * len(self.feat_conditions)

        for attr, attr_info in self.feat_conditions.items():
            if attr_info['type'] in (1, 2):
                inv_attr = self.dataset_info.dataset_description[attr]['original_position']
                gt, leq = attr_info['values']

                if gt is not None and (sample[inv_attr] < gt or abs(sample[inv_attr] - gt) < epsilon):
                    obs_def[attr] = [gt + epsilon]
                elif leq is not None and (sample[inv_attr] > leq or abs(sample[inv_attr] - leq) < epsilon):
                    obs_def[attr] = [leq - epsilon]
                else:
                    obs_def[attr] = [sample[inv_attr]]

            elif attr_info['type'] == 3:
                inv_attr = self.dataset_info.dataset_description[attr]['original_position']
                current_value = int(sample[inv_attr])

                if current_value not in attr_info['values']:
                    obs_def[attr] = [attr_info['values'][0]]
                else:
                    obs_def[attr] = [current_value]
            else:
                current_category = next(i for i in
                                        self.dataset_info.dataset_description[attr]['categories_original_position']
                                        if sample[i] > 0)
                valid_attrs = attr_info['values']

                if current_category in valid_attrs:
                    obs_def[attr] = [current_category]
                else:
                    obs_def[attr] = valid_attrs

        valid_obs = product(*obs_def)
        all_obs = np.zeros((len(obs_def), len(sample)))

        for i, obs in enumerate(obs_def):
            for feat_idx, value in enumerate(obs):
                attr_info = self.feat_conditions[i]
                if attr_info['type'] in (1, 2, 3):
                    inv_attr = self.dataset_info.dataset_description[attr]['original_position']
                    all_obs[i, inv_attr] = value
                else:
                    all_obs[i, value] = 1

        return all_obs
