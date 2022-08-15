# splitter.py
# from cython.parallel import parallel, prange
# from cython.operator cimport dereference as deref, preincrement as inc
import math
from typing import List, Tuple
from .extraction_problem import ExtractionContext, FeatureConditions, FeatureBounds, ValueIdsRanges
from .extraction_problem import estimate_probability

from .splitter_header import PartitionProb, SplitPointScore, SplitPoint
from .extraction_problem_header import ConditionSide

from .observations import CATEGORY_ACTIVATED, CATEGORY_DEACTIVATED, CATEGORY_UNKNOWN, CATEGORY_UNSET, ACTIVATED_MASK
from .observations import set_category_state, is_category_state


def agg_partition_prob(global_state: ExtractionContext, partition_prob: PartitionProb, rule: int, add: bool) -> None:
    mult: int = 1 if add else -1
    tree_id: int = global_state.problem.rule_tree[rule]

    tree_prob_sum: List[float] = partition_prob.tree_prob_sum[tree_id]  # [0]
    rule_prob: List[float] = global_state.problem.rule_probabilities[rule]  # [0]

    # prev_label_prob: float
    # prob_delta_tree: float

    previous_active_by_tree: int = partition_prob.active_by_tree[tree_id]

    if partition_prob.active_by_tree[tree_id] == 1 and mult == -1:
        print("Attempting to remove last rule...", rule, "from tree", tree_id)
        exit(0)

    partition_prob.active_by_tree[tree_id] += mult
    partition_prob.num_active += mult

    for i in range(global_state.problem.n_labels):
        if previous_active_by_tree != 0:
            prev_label_prob: float = tree_prob_sum[i] / previous_active_by_tree
        else:
            prev_label_prob: float = 0

        tree_prob_sum[i] += mult * rule_prob[i]

        prob_delta_tree = (tree_prob_sum[i] / partition_prob.active_by_tree[tree_id]) - prev_label_prob
        partition_prob.prob[i] += prob_delta_tree / global_state.problem.n_trees


def batch_set_values_bucket_state(global_state: ExtractionContext, partition: PartitionProb, condition_side: ConditionSide, bucket_from: int, bucket_to: int, state: bool) -> int:
    num_active: int = 0
    # rule_id: int
    # rule_pos: int
    # List[int].iterator it = condition_side.ids.begin() + condition_side.ids_values[bucket_from].ids_starts_at
    # List[int].iterator end = condition_side.ids.begin() + condition_side.ids_values[bucket_to].ids_ends_at

    if bucket_from <= bucket_to:
        start_pos: int = condition_side.ids_values[bucket_from].ids_starts_at
        end_pos: int = condition_side.ids_values[bucket_to].ids_ends_at

        # while it != end:
        #     rule_id = deref(it)
        #     inc(it)
        while start_pos != end_pos:
            rule_id: int = condition_side.ids[start_pos]
            start_pos += 1
            if global_state.active_rules[rule_id]:
                agg_partition_prob(global_state, partition, rule_id, state)  # ! ERROR PATH
                num_active += 1

    return num_active


def set_values_bucket_state(global_state: ExtractionContext, partition: PartitionProb, condition_side: ConditionSide,
                            bucket: int, state: bool) -> int:

    return batch_set_values_bucket_state(global_state, partition, condition_side, bucket, bucket, state)  # ! ERROR PATH


def compute_gain(partition_meet: PartitionProb, partition_not_meet: PartitionProb,
                 global_score: float) -> float:
    # Total number of rules. This does not represent the global number of rules, as there might be
    # rules that are in both partitions, so the total is likely higher than the global number of rules
    total: int = partition_not_meet.num_active + partition_meet.num_active
    meet_proportion: float = (partition_meet.num_active) / total
    not_meet_proportion: float = 1 - meet_proportion
    result: float = global_score - (meet_proportion * criterion(partition_meet.prob) +
                                    not_meet_proportion * criterion(partition_not_meet.prob))
    return result


def calculate_category_score(global_state: ExtractionContext, conditions: FeatureConditions,
                             partition_meet: PartitionProb, partition_not_meet: PartitionProb,
                             global_score: float, category: int) -> float:
    # Activate the 1 rules for the category (remember, currently all is set to 0s)
    num_activated: int = set_values_bucket_state(global_state, partition_meet, conditions.side_2, category, True)

    # Deactivate the 0 rules for the category
    set_values_bucket_state(global_state, partition_meet, conditions.side_1, category, False)

    # Deactivate the 1 rules for the category in the not meet partition, in this partition, both 1s and 0s rules
    # might be activated. Therefore, by deactivating the 1s rules, only the 0s rules remains
    set_values_bucket_state(global_state, partition_not_meet, conditions.side_2, category, False)
    return compute_gain(partition_meet, partition_not_meet, global_score)


def make_partition_prob(global_state: ExtractionContext, global_prob: List[float]) -> PartitionProb:
    tree_prob_sum: List[List[float]] = global_state.tree_prob_sum
    active_by_tree: List[int] = global_state.active_by_tree
    prob: List[float] = global_prob
    num_active = global_state.num_active_rules
    partition_prob: PartitionProb = PartitionProb(tree_prob_sum, active_by_tree, prob, num_active)
    return partition_prob


def categorical_max_split(global_state: ExtractionContext, feature: int, verbose: bool) -> SplitPointScore:
    global_prob: List[float] = estimate_probability(global_state)
    conditions: FeatureConditions = global_state.feature_conditions_view[feature]
    feature_bounds: FeatureBounds = global_state.feature_bounds[feature]
    # assert conditions.feature_type == 4, "Categorical split is only valid for one-hot encoded variables"
    num_categories: int = len(conditions.side_1.ids_values)
    categorical_mask: int = feature_bounds.categorical_mask

    # Partition that represents the current state
    partition_not_meet: PartitionProb = make_partition_prob(global_state, global_prob)

    # In this partition, all categories are negated (in one-hot, this would be all 0s, no categories activated)
    partition_meet: PartitionProb = make_partition_prob(global_state, global_prob)

    # Compute global score (needed to compute the uncertainty reduction)
    global_score: float = criterion(partition_meet.prob)

    # Setting all categories to 0s in the meet partition
    i: int = 0
    deactivated: int
    for i in range(num_categories):
        if is_category_state(categorical_mask, i, CATEGORY_UNSET):
            # Deactivate all rules that assert a column (disable all 1s in the one-hot)
            deactivated = set_values_bucket_state(global_state, partition_meet, conditions.side_2, i, False)
        elif is_category_state(categorical_mask, i, CATEGORY_ACTIVATED):
            print("Attempting to deactivate a category which was previously activated")
            exit(0)

    # Calculate split gain for each available category
    best_category: int = -1
    best_score: float = -1
    score: float
    num_active_categories: int = 0
    is_split_ok: bool = True

    for i in range(num_categories):
        if is_category_state(categorical_mask, i, CATEGORY_UNSET):
            score = calculate_category_score(
                global_state, global_state.feature_conditions_view[feature], partition_meet, partition_not_meet, global_score, i)
            num_active_categories += 1

            if best_score == -1 or score > best_score:
                best_score = score
                best_category = i

    # Bounds for the split
    new_categorical_mask_meet: int = 0x5555555555555555  # All disabled
    new_categorical_mask_not_meet: int = categorical_mask

    # Activate category in meet split, and disable in not met split
    new_categorical_mask_meet = set_category_state(new_categorical_mask_meet, best_category, CATEGORY_ACTIVATED)
    new_categorical_mask_not_meet = set_category_state(
        new_categorical_mask_not_meet, best_category, CATEGORY_DEACTIVATED)

    # If there is only two categories activated, then the remaining category should be activated in the not
    # meet side
    if num_active_categories == 2:
        new_categorical_mask_not_meet = 0x5555555555555555  # All disabled
        for i in range(num_categories):
            if is_category_state(categorical_mask, i, CATEGORY_UNSET) and i != best_category:
                new_categorical_mask_not_meet = set_category_state(new_categorical_mask_not_meet, i, CATEGORY_ACTIVATED)
                break

    elif num_active_categories == 0:
        feature_bounds.is_mask_set = True
        is_split_ok = False

    # Create the new bounds
    zero_pair: List[int] = [0, 0]
    new_bounds_meet: FeatureBounds = make_feature_bounds(zero_pair, new_categorical_mask_meet, True)
    new_bounds_not_meet: FeatureBounds = make_feature_bounds(
        zero_pair, new_categorical_mask_not_meet, num_active_categories == 2)
    split_point: SplitPoint = make_split_point(best_category, feature, True, new_bounds_meet, new_bounds_not_meet)
    return make_split_point_score(best_score, split_point, is_split_ok)


def make_split_point_score(score: float, split_point: SplitPoint, is_ok: bool) -> SplitPointScore:
    split_score: SplitPointScore = SplitPointScore(score, split_point, is_ok)
    # split_score.score = score
    # split_score.split_point = split_point
    # split_score.is_ok = is_ok
    return split_score


def make_split_point(value: float, feature: int, meet: bool, bmeet: FeatureBounds, bnot_meet: FeatureBounds) -> SplitPoint:
    split_point: SplitPoint = SplitPoint(value, feature, meet, bmeet, bnot_meet)
    # split_point.value = value
    # split_point.feature = feature
    # split_point.meet = meet
    # split_point.bounds_meet = bmeet
    # split_point.bounds_not_meet = bnot_meet
    return split_point


def make_feature_bounds(bounds: List[int], categorical_mask: int, is_mask_set: bool) -> FeatureBounds:
    feature_bounds: FeatureBounds = FeatureBounds(bounds, categorical_mask, is_mask_set)
    # feature_bounds.numerical_bounds = bounds
    # feature_bounds.categorical_mask = categorical_mask
    # feature_bounds.is_mask_set = is_mask_set
    return feature_bounds


def numerical_max_split(global_state: ExtractionContext, feature: int) -> SplitPointScore:
    global_prob: List[float] = estimate_probability(global_state)
    conditions: FeatureConditions = global_state.feature_conditions_view[feature]
    feature_bounds: FeatureBounds = global_state.feature_bounds[feature]

    # Partition for rules that meet feat > x
    partition_not_meet: PartitionProb = make_partition_prob(global_state, global_prob)

    # Partition for rules that meet feat <= x
    partition_meet: PartitionProb = make_partition_prob(global_state, global_prob)

    # Compute global score (needed to compute the uncertainty reduction)
    global_score: float = criterion(partition_meet.prob)

    min_bucket: int = feature_bounds.numerical_bounds[0]
    max_bucket: int = feature_bounds.numerical_bounds[1]
    current_bucket: int = min_bucket

    # Deactivate greater than conditions (side_2) in the meet partition, because at the beginning,
    # the partition starts with the rules which meet the condition feat <= min(x), thus, all greater than
    # conditions are deactivated
    for bucket in range(min_bucket, max_bucket):
        set_values_bucket_state(global_state, partition_meet, conditions.side_2, bucket, False)

    # In this iteration, the partition meet holds all lte rules and does not have any gt
    # since the threshold is the smallest (rules defined in <= smallest).
    # In the not meet partition, we deactivate the rules whose threshold is == smallest, since this partition
    # contains all rules defined in > threshold (and <= threshold is not)
    set_values_bucket_state(global_state, partition_not_meet, conditions.side_1, current_bucket, False)

    current_score: float = compute_gain(partition_meet, partition_not_meet, global_score)
    best_threshold: float = conditions.side_1.ids_values[current_bucket].value
    best_score: float = current_score
    best_position: int = current_bucket
    current_bucket += 1

    # num_changed: int

    while current_bucket < max_bucket:
        # Notice that, a bucket i refers to the same threshold in side_1 and side_2
        # for simplicity, we refer to these number as threshold in the next comments

        # meet partition: all rules that satisfy rule <= threshold (all lte rules, gt rules that threshold(rule) < threshold
        # not meet partition: all rules that satisfy rule > threshold (all gt rules, lte rules that threshold(rule) > threshold)
        num_changed: int = set_values_bucket_state(
            global_state, partition_meet, conditions.side_2, current_bucket - 1, True)

        # Deactivating lte rules with rule(threshold) < threshold, because there are no longer greater than threshold
        # since this partition know only holds rules that meet rule > threshold
        num_changed += set_values_bucket_state(global_state, partition_not_meet,
                                               conditions.side_1, current_bucket, False)  # ! ERROR PATH

        if num_changed > 0:
            # Compute the current gain for the split
            current_gain = compute_gain(partition_meet, partition_not_meet, global_score)

            if current_gain > best_score:
                best_score = current_gain
                best_threshold = conditions.side_1.ids_values[current_bucket].value
                best_position = current_bucket

        current_bucket += 1

    new_bounds_meet: FeatureBounds = make_feature_bounds([min_bucket, best_position], 0, False)
    new_bounds_not_meet: FeatureBounds = make_feature_bounds([best_position + 1, max_bucket], 0, False)
    split_point: SplitPoint = make_split_point(best_threshold, feature, True, new_bounds_meet, new_bounds_not_meet)
    return make_split_point_score(best_score, split_point, True)


def calculate_feature_split(global_state: ExtractionContext, feature: int, verbose: bool) -> SplitPointScore:
    feature_conditions: FeatureConditions = global_state.feature_conditions_view[feature]

    if feature_conditions.feature_type == 4:
        # One-hot
        return categorical_max_split(global_state, feature, verbose)
    else:
        # Real, ordinal and binary
        return numerical_max_split(global_state, feature)  # ! ERROR PATH


def calculate_split(global_state: ExtractionContext, verbose: bool = False) -> Tuple[bool, SplitPoint]:
    # current_split: SplitPointScore
    # feature_bounds: FeatureBounds
    # feature_type: int

    mock_feature_bounds: FeatureBounds = make_feature_bounds([0, 0], 0, False)
    mock_split_point: SplitPoint = make_split_point(0, -1, False, mock_feature_bounds, mock_feature_bounds)
    best_split: SplitPointScore = make_split_point_score(-1, mock_split_point, False)

    n_features = global_state.problem.n_features
    # feature: int

    for feature in range(n_features):
        feature_bounds: FeatureBounds = global_state.feature_bounds[feature]
        feature_type: int = global_state.feature_conditions_view[feature].feature_type

        if (feature_type == 4 and not feature_bounds.is_mask_set) or (feature_type != 4 and feature_bounds.numerical_bounds[0] < feature_bounds.numerical_bounds[1]):

            current_split: SplitPointScore = calculate_feature_split(global_state,  feature, verbose)  # ! ERROR PATH

            if not best_split.is_ok or (current_split.is_ok and current_split.score > best_split.score):
                best_split = current_split

    return (best_split.is_ok, best_split.split_point)

# cdef float gini(List[float] & label_probs) :
#         cdef float score = 0
#         for prob in label_probs:
#             score += prob * prob
#         return 1 - score

# Shannon entropy


def criterion(label_probs: List[float]) -> float:
    score: float = 0
    for prob in label_probs:
        if prob > 0:
            score += prob * math.log2(prob)
    return -score
