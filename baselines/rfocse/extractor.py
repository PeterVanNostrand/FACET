# from cython.operator import dereference
from math import floor
from typing import List, Tuple
from .extraction_problem import ExtractionContext, ConditionSide, FeatureConditions, FeatureBounds, SplitPoint, Solution
from .extraction_problem import estimate_probability
from .extractor_header import StepState
from .math_utils import float_argmax, sum_vector, sub_vector
from .observations import is_category_state, set_category_state, does_meet_split, update_to_meet_split
from .observations import CATEGORY_ACTIVATED, CATEGORY_DEACTIVATED, CATEGORY_UNSET, CATEGORY_UNKNOWN
from .splitter import calculate_split
from .observations import distance as rule_distance
from .debug import LOG_LEVEL, ExtractionContext_tostr, Solution_tostr, instance_num
import copy


def debug(global_state: ExtractionContext) -> None:
    print("Num activate rules", global_state.num_active_rules,
          "Estimated probability ", estimate_probability(global_state))
    print(global_state.active_by_tree)
    print(global_state.tree_prob_sum)
    print(get_set_rules_ids(global_state))

    print("Split dump")
    for split_point in global_state.splits:
        print(split_point.feature, split_point.meet, split_point.value)

    print("Feature bounds dump")
    iz = 0
    for feature_bound in global_state.feature_bounds:
        print(iz, feature_bound.numerical_bounds, feature_bound.categorical_mask, feature_bound.is_mask_set)
        iz += 1


def activate_rule(global_state: ExtractionContext, rule_id: int) -> bool:
    if global_state.active_rules[rule_id]:
        raise ValueError()

    if global_state.rule_blacklist[rule_id]:
        # This rule has been previously filtered
        return False

    tree_id: int = global_state.problem.rule_tree[rule_id]

    global_state.active_rules[rule_id] = True
    global_state.num_active_rules += 1

    global_state.active_by_tree[tree_id] += 1
    sum_vector(global_state.problem.rule_probabilities[rule_id], global_state.tree_prob_sum[tree_id])
    return True


def deactivate_rule(global_state: ExtractionContext, rule_id: int) -> None:
    if not global_state.active_rules[rule_id]:
        raise ValueError()

    tree_id: int = global_state.problem.rule_tree[rule_id]

    if global_state.active_by_tree[tree_id] == 1:
        print("Trying to remove last rule from tree", tree_id, "rule", rule_id)
        debug(global_state)
        exit(0)

    global_state.active_rules[rule_id] = False
    global_state.num_active_rules -= 1

    global_state.active_by_tree[tree_id] -= 1
    sub_vector(global_state.problem.rule_probabilities[rule_id], global_state.tree_prob_sum[tree_id])


def early_stop(global_state: ExtractionContext) -> int:
    num_set: int = 0
    set_probs: List[float] = [0.0 for _ in range(global_state.problem.n_labels)]
    t_id: int = 0
    l: int = 0
    non_zero: int = -1

    for t_id in range(global_state.problem.n_trees):
        non_zero = -1
        for l in range(global_state.problem.n_labels):
            if global_state.tree_prob_sum[t_id][l] > 0:
                if non_zero == -1:
                    non_zero = l
                else:
                    non_zero = -2
                    break

        if non_zero > -1:
            set_probs[non_zero] += 1
            num_set += 1

    max_label: int = float_argmax(set_probs)
    num_unset: int = global_state.problem.n_trees - num_set

    worst_case_prob: float = 1.0 * num_unset

    for l in range(global_state.problem.n_labels):
        if l != max_label and set_probs[max_label] <= (set_probs[l] + worst_case_prob):
            return -1

    return max_label


def can_stop(global_state: ExtractionContext) -> int:
    labels_probs: List[float]
    rule_prob: List[float]

    estimated_class: int = -1

    if global_state.num_active_rules == global_state.problem.n_trees:
        labels_probs: List[float] = [0.0 for _ in range(global_state.problem.n_labels)]

        for i in range(len(global_state.active_rules)):
            if global_state.active_rules[i]:
                rule_prob = global_state.problem.rule_probabilities[i]

                for l in range(global_state.problem.n_labels):
                    labels_probs[l] += rule_prob[l]

        estimated_class = float_argmax(labels_probs)
    else:
        estimated_class = early_stop(global_state)

    return estimated_class


def is_bucket_active(global_state: ExtractionContext, condition_side: ConditionSide, bucket: int) -> bool:
    for rule_pos in range(condition_side.ids_values[bucket].ids_starts_at,
                          condition_side.ids_values[bucket].ids_ends_at):
        rule_id = condition_side.ids[rule_pos]
        if global_state.active_rules[rule_id] and not global_state.rule_blacklist[rule_id]:
            return True
    return False


def update_bounds_categorical(global_state: ExtractionContext, feature: int) -> None:
    feature_bounds: FeatureBounds = global_state.feature_bounds[feature]
    feature_conditions: FeatureConditions = global_state.feature_conditions_view[feature]
    num_categories: int
    num_categories = len(feature_conditions.side_1.ids_values)
    for bucket in range(num_categories):
        if is_category_state(feature_bounds.categorical_mask, bucket, CATEGORY_UNSET) and \
                not (is_bucket_active(global_state, feature_conditions.side_1, bucket) and
                     is_bucket_active(global_state, feature_conditions.side_2, bucket)):

            feature_bounds.categorical_mask = set_category_state(feature_bounds.categorical_mask,
                                                                 bucket,
                                                                 CATEGORY_UNKNOWN)


def soft_numerical_pruning(global_state: ExtractionContext, feature: int, solution: Solution, removed_rules: List[int], deactivate: bool = True, feature_only_dist: bool = False) -> None:
    feature_bounds: FeatureBounds = global_state.feature_bounds[feature]
    feature_conditions: FeatureConditions = global_state.feature_conditions_view[feature]
    side_1: ConditionSide = feature_conditions.side_1
    side_2: ConditionSide = feature_conditions.side_2
    min_bucket: int = feature_bounds.numerical_bounds[0]
    max_bucket: int = feature_bounds.numerical_bounds[1]
    current_solution_distance: float = solution.distance if solution.found else global_state.max_distance
    i: int
    factual_value: float = global_state.to_explain[feature]
    current_value: float = global_state.current_obs[feature]

    max_bucket -= 1
    updated_value: float

    # Min bucket
    while min_bucket <= max_bucket:
        if feature_conditions.feature_type == 1:
            updated_value = side_1.ids_values[min_bucket].value
        else:
            updated_value = floor(side_1.ids_values[min_bucket].value)

        global_state.current_obs[feature] = updated_value
        if factual_value <= updated_value or \
                calculate_current_distance(global_state) <= current_solution_distance:
            break

        if deactivate:
            deactivate_bucket(global_state, side_1, min_bucket, removed_rules)

        min_bucket += 1

    # Max bucket
    while min_bucket <= max_bucket:
        if feature_conditions.feature_type == 1:
            updated_value = side_1.ids_values[max_bucket].value
        else:
            updated_value = floor(side_1.ids_values[max_bucket].value + 1)

        global_state.current_obs[feature] = updated_value
        if factual_value >= updated_value or\
                calculate_current_distance(global_state) <= current_solution_distance:
            break

        if deactivate:
            deactivate_bucket(global_state, side_2, max_bucket, removed_rules)
        max_bucket -= 1

    global_state.current_obs[feature] = current_value

    if max_bucket < min_bucket:
        feature_bounds.numerical_bounds[0] = min_bucket
        feature_bounds.numerical_bounds[1] = min_bucket
    else:
        feature_bounds.numerical_bounds[0] = min_bucket
        feature_bounds.numerical_bounds[1] = max_bucket + 1


def update_bounds_numerical(global_state: ExtractionContext, feature: int) -> None:
    feature_bounds: FeatureBounds = global_state.feature_bounds[feature]
    feature_conditions: FeatureConditions = global_state.feature_conditions_view[feature]
    side_1: ConditionSide = feature_conditions.side_1
    side_2: ConditionSide = feature_conditions.side_2
    min_bucket: int = feature_bounds.numerical_bounds[0]
    max_bucket: int = feature_bounds.numerical_bounds[1]
    max_bucket -= 1

    while min_bucket <= max_bucket and not is_bucket_active(global_state, feature_conditions.side_1, min_bucket):
        min_bucket += 1

    while min_bucket <= max_bucket and not is_bucket_active(global_state, feature_conditions.side_2, max_bucket):
        max_bucket -= 1

    if max_bucket < min_bucket:
        feature_bounds.numerical_bounds[0] = min_bucket
        feature_bounds.numerical_bounds[1] = min_bucket
    else:
        feature_bounds.numerical_bounds[0] = min_bucket
        feature_bounds.numerical_bounds[1] = max_bucket + 1


def soft_pruning(global_state: ExtractionContext, solution: Solution, removed_rules: List[int]) -> None:
    feature_conditions: FeatureConditions

    for feature in range(global_state.problem.n_features):
        feature_conditions = global_state.feature_conditions_view[feature]
        if feature_conditions.feature_type != 4:
            soft_numerical_pruning(global_state, feature, solution, removed_rules)


def ensure_bounds_consistency(global_state: ExtractionContext, solution: Solution) -> None:
    feature: int = 0
    feature_conditions: FeatureConditions = global_state.feature_conditions_view[feature]
    removed: List[int] = []

    for feature in range(global_state.problem.n_features):
        if feature_conditions.feature_type == 4:
            update_bounds_categorical(global_state, feature)
        else:
            update_bounds_numerical(global_state, feature)
            assert len(removed) == 0


def deactivate_bucket_batch(global_state: ExtractionContext, condition_side: ConditionSide, bucket_from: int, bucket_to: int, removed_rules: List[int]) -> None:
    rule_id: int
    rule_pos: int
    if bucket_from <= bucket_to:
        for rule_pos in range(condition_side.ids_values[bucket_from].ids_starts_at,
                              condition_side.ids_values[bucket_to].ids_ends_at):
            rule_id = condition_side.ids[rule_pos]

            if global_state.active_rules[rule_id]:
                deactivate_rule(global_state, rule_id)
                removed_rules.append(rule_id)


def deactivate_bucket(global_state: ExtractionContext, condition_side: ConditionSide, bucket: int, removed_rules: List[int]) -> None:
    deactivate_bucket_batch(global_state, condition_side, bucket, bucket, removed_rules)


def apply_feature_bounds_categorical(global_state: ExtractionContext, split_point: SplitPoint) -> StepState:
    removed_rules: List[int]
    previous_feature_bounds: FeatureBounds = global_state.feature_bounds[split_point.feature]
    feature_conditions: FeatureConditions = global_state.feature_conditions_view[split_point.feature]
    num_categories: int = len(feature_conditions.side_1.ids_values)

    feature_bounds: FeatureBounds = split_point.bounds_meet if split_point.meet else split_point.bounds_not_meet

    previous_mask: int = previous_feature_bounds.categorical_mask
    current_mask: int = feature_bounds.categorical_mask

    for category in range(num_categories):
        if is_category_state(previous_mask, category, CATEGORY_UNSET) and \
                is_category_state(current_mask, category, CATEGORY_DEACTIVATED):
            # Category was previously inactive and now it is being negated
            deactivate_bucket(global_state, feature_conditions.side_2, category, removed_rules)
            pass
        elif is_category_state(previous_mask, category, CATEGORY_UNSET) and \
                is_category_state(current_mask, category, CATEGORY_ACTIVATED):
            # Category is activated, so we disable the rules that negate the category
            deactivate_bucket(global_state, feature_conditions.side_1, category, removed_rules)
        else:
            # Nothing to do:
            # Category was previously deactivated (previous_category_state == -1) or is unset
            pass

    state: StepState = StepState(removed_rules, copy.deepcopy(global_state.feature_bounds))
    global_state.feature_bounds[split_point.feature] = feature_bounds
    return state


def apply_feature_bounds_numerical(global_state: ExtractionContext, split_point: SplitPoint) -> StepState:
    removed_rules: List[int] = []
    previous_feature_bounds: FeatureBounds = global_state.feature_bounds[split_point.feature]
    feature_conditions: FeatureConditions = global_state.feature_conditions_view[split_point.feature]

    previous_min_bucket: int = previous_feature_bounds.numerical_bounds[0]
    previous_max_bucket: int = previous_feature_bounds.numerical_bounds[1]

    feature_bounds: FeatureBounds = split_point.bounds_meet if split_point.meet else split_point.bounds_not_meet
    feature_bounds_not: FeatureBounds = split_point.bounds_meet if not split_point.meet else split_point.bounds_not_meet

    current_min_bucket: int = feature_bounds.numerical_bounds[0]
    current_max_bucket: int = feature_bounds.numerical_bounds[1]

    deactivate_bucket_batch(global_state, feature_conditions.side_2,
                            current_max_bucket, previous_max_bucket - 1, removed_rules)
    deactivate_bucket_batch(global_state, feature_conditions.side_1,
                            previous_min_bucket, current_min_bucket - 1, removed_rules)

    state: StepState = StepState(removed_rules, copy.deepcopy(global_state.feature_bounds))
    global_state.feature_bounds[split_point.feature] = feature_bounds
    return state


def apply_feature_bounds(global_state: ExtractionContext, solution: Solution, split_point: SplitPoint) -> StepState:
    feature_conditions: FeatureConditions = global_state.feature_conditions_view[split_point.feature]
    state: StepState

    if feature_conditions.feature_type == 4:
        state = apply_feature_bounds_categorical(global_state, split_point)
    else:
        state = apply_feature_bounds_numerical(global_state, split_point)

    global_state.splits.append(split_point)
    soft_pruning(global_state, solution, state.removed_rules)
    ensure_bounds_consistency(global_state, solution)
    return state


def rollback_split(global_state: ExtractionContext, step_state: StepState) -> None:
    global_state.feature_bounds = step_state.previous_bounds
    global_state.splits.pop()
    for rule_id in step_state.removed_rules:
        activate_rule(global_state, rule_id)


def get_set_trees_ids(global_state: ExtractionContext) -> List[int]:
    set_rules: List[int] = []
    t_id: int
    for r_id in range(global_state.problem.n_rules):
        t_id = global_state.problem.rule_tree[r_id]
        if global_state.active_by_tree[t_id] == 1 and global_state.active_rules[r_id]:
            set_rules.append(r_id)
    return set_rules


def get_set_rules_ids(global_state: ExtractionContext) -> List[int]:
    set_rules: List[int] = []
    t_id: int
    for r_id in range(global_state.problem.n_rules):
        if global_state.active_rules[r_id]:
            set_rules.append(r_id)
    return set_rules


def calculate_current_distance(global_state: ExtractionContext) -> float:
    return rule_distance(global_state.problem, global_state.to_explain,
                         global_state.current_obs)


def prune_rules(global_state: ExtractionContext, solution: Solution) -> bool:
    current_solution_distance: float = solution.distance if solution.found else global_state.max_distance

    if current_solution_distance < 0:
        # A solution has not been found yet
        return False

    current_rule_pruning: int = global_state.current_rule_pruning
    rule_id: int
    while current_rule_pruning > 0 and \
            global_state.sorted_rule_distances[current_rule_pruning][1] > current_solution_distance:

        rule_id = global_state.sorted_rule_distances[current_rule_pruning][0]
        t_id = global_state.problem.rule_tree[rule_id]
        global_state.rule_blacklist[rule_id] = True

        if global_state.active_rules[rule_id]:
            deactivate_rule(global_state, rule_id)

        current_rule_pruning -= 1

    changed: bool = global_state.current_rule_pruning != current_rule_pruning
    global_state.current_rule_pruning = current_rule_pruning
    return changed


def extract_counterfactual_impl(global_state: ExtractionContext, solution: Solution) -> None:
    solution.num_evaluations += 1
    iteration: int = solution.num_evaluations

    global instance_num
    if LOG_LEVEL == "DEBUG":
        f = open("logs/py_log_{}.txt".format(instance_num), "a")
        f.write("global_state (extract_counterfactual_impl) in START:\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global_state (extract_counterfactual_impl) in END:\n")
        f.write("solution (extract_counterfactual_impl) in START:\n")
        f.write(Solution_tostr(solution))
        f.write("solution (extract_counterfactual_impl) in END:\n")
        f.write("CHECKING CONDITIONS\n")
        f.close()

    if (solution.found and not global_state.problem.search_closest) \
            or (global_state.problem.max_iterations != -1 and iteration >= global_state.problem.max_iterations):
        return

    if LOG_LEVEL == "DEBUG":
        f = open("logs/py_log_{}.txt".format(instance_num), "a")
        f.write("solution:\n")
        f.write(Solution_tostr(solution))
        f.write("PRUNING RULES\n")
        f.close()

    if prune_rules(global_state, solution) or solution.num_evaluations == 1:
        ensure_bounds_consistency(global_state, solution)

    if LOG_LEVEL == "DEBUG":
        f = open("logs/py_log_{}.txt".format(instance_num), "a")
        f.write("solution:\n")
        f.write(Solution_tostr(solution))
        f.close()

    estimated_class: int = can_stop(global_state)

    step_state: StepState
    previous_value: float

    split_point: SplitPoint
    split_point_ok: Tuple[bool, SplitPoint]

    if iteration > 0 and global_state.problem.log_every > 0 and (iteration % global_state.problem.log_every) == 0:
        print("Iteration number [{:d}]. Discovered foil [{:d}]. Current solution distance [{:.06f}]. Current distance [{:.06f}]. Pruned rules [{:d}] out of [{:d}]. Max distance [{:.06f}]\n".format(
            iteration,
            solution.num_discovered_non_foil,
            solution.distance,
            calculate_current_distance(global_state),
            global_state.problem.n_rules - global_state.current_rule_pruning - 1,
            global_state.problem.n_rules,
            global_state.max_distance
        ))

    if LOG_LEVEL == "DEBUG":
        f = open("logs/py_log_{}.txt".format(instance_num), "a")
        f.write("global_state:\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("solution:\n")
        f.write(Solution_tostr(solution))
        f.write("estimated_class: " + str(estimated_class) + "\n")
        f.close()

    if estimated_class != -1:

        if estimated_class == global_state.real_class:
            solution.num_discovered_non_foil += 1
        else:
            solution.found = True
            solution.conditions: List[SplitPoint] = []
            for i in range(len(global_state.splits)):
                solution.conditions.append(copy.deepcopy(global_state.splits[i]))
            solution.set_rules = get_set_rules_ids(global_state)
            solution.distance = calculate_current_distance(global_state)
            solution.instance: List[float] = global_state.current_obs.copy()
            solution.estimated_class = estimated_class
    else:
        verbose = False
        split_point_ok = calculate_split(global_state, verbose)  # ! ERROR PATH

        if not split_point_ok[0]:
            print("INSTANCE: {}".format(instance_num))
            debug(global_state)
            exit(0)

        max_distance = solution.distance if solution.found else global_state.max_distance
        split_point = split_point_ok[1]
        split_point.meet = does_meet_split(global_state.problem, global_state.current_obs, split_point)
        previous_value = global_state.current_obs[split_point.feature]
        update_to_meet_split(global_state.problem, global_state.current_obs, split_point, True)

        if calculate_current_distance(global_state) < max_distance:
            previous_bounds = global_state.feature_bounds[split_point.feature]
            previous_rules = global_state.active_rules
            previous_split_values = global_state.feature_bounds[split_point.feature]
            step_state = apply_feature_bounds(global_state, solution, split_point)
            extract_counterfactual_impl(global_state, solution)
            rollback_split(global_state, step_state)

        global_state.current_obs[split_point.feature] = previous_value
        max_distance = solution.distance if solution.found else global_state.max_distance

        if not solution.found or global_state.problem.search_closest:
            previous_distance = calculate_current_distance(global_state)
            split_point.meet = not split_point.meet
            update_to_meet_split(global_state.problem, global_state.current_obs, split_point, True)

            if calculate_current_distance(global_state) < max_distance:
                previous_n_active_rules = global_state.num_active_rules
                step_state = apply_feature_bounds(global_state, solution, split_point)
                extract_counterfactual_impl(global_state, solution)
                rollback_split(global_state, step_state)

            global_state.current_obs[split_point.feature] = previous_value
