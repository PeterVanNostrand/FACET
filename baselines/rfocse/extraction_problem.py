from math import floor
from operator import itemgetter
from typing import List, Tuple

# from .observations import adapt_observation_representation
# from .observations cimport calculate_sorted_rule_distances
from .extraction_problem_header import *
from .math_utils import float_argmax, sum_vector
from .observations import adapt_observation_representation, calculate_sorted_rule_distances


def make_condition_side(condition_side, feature: int, feature_type: int, is_side_1: bool) -> ConditionSide:
    thresholds_ids = sorted(condition_side.items(), key=itemgetter(0))
    ids_values_c: List[ValueIdsRanges] = []
    # id_values_c: ValueIdsRanges
    # threshold_c: float
    # starts_at: int
    # ends_at: int
    all_ids = []
    join = False

    bucket_thresholds = list(enumerate(thresholds_ids))

    for i in range(len(thresholds_ids)):
        bucket_id, (threshold, ids) = bucket_thresholds[i]

        if not join:
            starts_at: int = len(all_ids)

        all_ids.extend(sorted(ids))
        ends_at: int = len(all_ids)

        if feature_type == 2 and floor(threshold) == threshold:
            if i + 1 < len(bucket_thresholds):
                _, (next_threshold, _) = bucket_thresholds[i + 1]

                if floor(next_threshold) == threshold:
                    join = True
                    continue

            threshold += 0.5

        join = False
        id_values_c: ValueIdsRanges = ValueIdsRanges(
            threshold,
            starts_at,
            ends_at,
            ends_at - starts_at
        )
        ids_values_c.append(id_values_c)

    ids_c: List[int] = all_ids

    return ConditionSide(
        ids_values_c,
        ids_c
    )


def make_problem(parsed_rf, dataset_description, search_closest_c: bool, log_every_c: int,
                 max_iterations_c: int) -> ExtractionProblem:
    feature_conditions_list_c: List[FeatureConditions] = []

    for feat, feat_info in dataset_description.dataset_description.items():
        conditions = parsed_rf.feats_conditions[feat]
        feature_type = feat_info['type']
        feature_range = feat_info['range']

        side_1 = make_condition_side(conditions['side_1'], feat, feature_type, True)
        side_2 = make_condition_side(conditions['side_2'], feat, feature_type, False)

        feature_conditions_c = FeatureConditions(
            feature_type,
            side_1,
            side_2,
            feature_range
        )
        feature_conditions_list_c.append(feature_conditions_c)

    rule_probabilities_c: List[List[float]] = parsed_rf.rule_probs
    rule_tree_mapping_c: List[int] = [parsed_rf.rule_tree[i] for i in range(len(parsed_rf.rule_tree))]
    n_features_c: int = len(dataset_description.dataset_description)
    n_trees_c: int = parsed_rf.n_trees
    n_labels_c: int = parsed_rf.n_labels
    n_rules_c: int = len(parsed_rf.rule_probs)

    problem: ExtractionProblem = ExtractionProblem(
        # problem.feature_conditions =
        feature_conditions_list_c,
        # problem.rule_probabilities =
        rule_probabilities_c,
        # problem.rule_tree =
        rule_tree_mapping_c,
        # problem.n_rules =
        n_rules_c,
        # problem.n_features =
        n_features_c,
        # problem.n_labels =
        n_labels_c,
        # problem.n_trees =
        n_trees_c,
        # problem.search_closest =
        search_closest_c,
        # problem.log_every =
        log_every_c,
        # problem.max_iterations =
        max_iterations_c)

    return problem


def optimize_condition_side(condition_side: ConditionSide, to_remove) -> ConditionSide:
    optimized_condition_side: ConditionSide = ConditionSide([], [])
    current_ids_ranges: ValueIdsRanges

    i: int = 0
    start: int
    end: int
    rule_id: int

    for current_ids_ranges in condition_side.ids_values:
        start = len(optimized_condition_side.ids)

        for i in range(current_ids_ranges.ids_starts_at, current_ids_ranges.ids_ends_at):
            rule_id = condition_side.ids[i]
            if rule_id not in to_remove:
                optimized_condition_side.ids.append(rule_id)

        end = len(optimized_condition_side.ids)
        new_ids_values = ValueIdsRanges(current_ids_ranges.value, start, end, end - start)
        optimized_condition_side.ids_values.append(new_ids_values)

    return optimized_condition_side


def early_rule_pruning(problem: ExtractionProblem, sorted_rule_distances: List[Tuple[int, float]], max_distance: float):
    current: Tuple[int, float]
    to_remove = set()

    for current in reversed(sorted_rule_distances):
        if current[1] <= (max_distance + 0.001):
            break

        to_remove.add(current[0])

    return to_remove


def estimate_min_distance(problem: ExtractionProblem, sorted_rule_distances: List[Tuple[int, float]], factual_class: int) -> float:
    set_trees: List[bool] = [False for _ in range(problem.n_trees)]
    n_set_trees: int = 0
    i: int = 0
    t_id: int
    current: Tuple[int, float]
    distance: float
    acc_prob: float = 0
    target_prob: float = len(set_trees) / 2  # Assume problem is binary
    while True:
        current = sorted_rule_distances[i]
        rule_class = float_argmax(problem.rule_probabilities[current[0]])
        t_id = problem.rule_tree[current[0]]
        if rule_class != factual_class and not set_trees[t_id]:
            set_trees[t_id] = True
            n_set_trees += 1
            distance = current[1]
            acc_prob += problem.rule_probabilities[current[0]][rule_class]

        if n_set_trees == len(set_trees) or acc_prob > target_prob:
            break
        i += 1

    return distance


def optimize_feature_conditions_storage(problem: ExtractionProblem, to_remove) -> List[FeatureConditions]:

    optimized_feature_conditions: List[FeatureConditions] = []
    # current_feature_conditions: FeatureConditions
    # i: int

    for i in range(len(problem.feature_conditions)):
        current_feature_conditions: FeatureConditions = problem.feature_conditions[i]

        optimized_feature_conditions.append(FeatureConditions(
            current_feature_conditions.feature_type,
            optimize_condition_side(current_feature_conditions.side_1, to_remove),
            optimize_condition_side(current_feature_conditions.side_2, to_remove),
            current_feature_conditions.feat_range,
        ))

    return optimized_feature_conditions


def create_extraction_state(problem: ExtractionProblem, parsed_rf, factual_class, observation,
                            dataset_info, max_distance) -> ExtractionContext:
    active_rules: List[bool] = [True for _ in range(problem.n_rules)]
    feature_bounds: List[FeatureBounds] = []
    tree_prob_sum: List[List[float]] = []
    active_by_tree: List[int] = [0 for _ in range(problem.n_trees)]
    splits: List[SplitPoint] = []
    max_distance_c: float = max_distance
    rule_backlist: List[bool] = [False for _ in range(problem.n_rules)]
    current_obs: List[float] = adapt_observation_representation(observation, dataset_info,
                                                                include_all_categories=False)
    to_explain: List[float] = adapt_observation_representation(observation, dataset_info,
                                                               include_all_categories=False)

    sorted_rule_distances: List[Tuple[int, float]] = calculate_sorted_rule_distances(problem, observation,
                                                                                     parsed_rf,
                                                                                     dataset_info)

    to_remove_rules = early_rule_pruning(problem, sorted_rule_distances, max_distance_c)
    sorted_rule_end_idx: int = problem.n_rules - len(to_remove_rules)
    feature_conditions_view: List[FeatureConditions] = optimize_feature_conditions_storage(
        problem, to_remove_rules)

    min_possible_distance: float = estimate_min_distance(problem, sorted_rule_distances, factual_class)

    for _ in range(problem.n_trees):
        tree_prob_sum.append([0.0 for _ in range(problem.n_labels)])

    num_active_rules: int = 0
    for i in range(problem.n_rules):
        if i not in to_remove_rules:
            num_active_rules += 1
            tree = problem.rule_tree[i]
            active_by_tree[tree] += 1
            sum_vector(problem.rule_probabilities[i], tree_prob_sum[tree])
            active_rules[i] = True
            rule_backlist[i] = False

        else:
            tree = problem.rule_tree[i]
            active_rules[i] = False
            rule_backlist[i] = True

    for feat in range(problem.n_features):
        feat_conds = feature_conditions_view[feat]
        feature_bounds.append(FeatureBounds([0, len(feat_conds.side_1.ids_values)], 0, False))

    return ExtractionContext(
        # Problem problem =
        problem,
        # List[bool] active_rules  =
        active_rules,
        # List[FeatureBounds] feature_bounds =
        feature_bounds,
        # List[FeatureConditions] feature_conditions_view =
        feature_conditions_view,
        # List[List[float]] tree_prob_sum =
        tree_prob_sum,
        # List[int] active_by_tree =
        active_by_tree,
        # List[SplitPoint] splits =
        splits,
        # int num_active_rules =
        num_active_rules,
        # List[float] current_obs =
        current_obs,
        # List[float] to_explain =
        to_explain,
        # int real_class =
        factual_class,
        # List[cpppair[int, float]] sorted_rule_distances =
        sorted_rule_distances,
        # int current_rule_filter,
        problem.n_rules - 1,
        # List[bool] rule_blackList,
        rule_backlist,
        # float max_distance =
        max_distance_c,
        # int sorted_rule_end_idx =
        sorted_rule_end_idx,
        # bool debug =
        False
    )


def estimate_probability(global_state: ExtractionContext) -> List[float]:
    current_prob: List[float] = [0.0 for _ in range(global_state.problem.n_labels)]
    # tree_prob: float, was used for pointer arithmetic, removed
    # active_by_tree: int

    for i in range(global_state.problem.n_trees):
        active_by_tree: int = global_state.active_by_tree[i]
        tree_prob = global_state.tree_prob_sum[i]  # [0]

        for l in range(global_state.problem.n_labels):
            current_prob[l] += tree_prob[l] / (active_by_tree * global_state.problem.n_trees)

    return current_prob
