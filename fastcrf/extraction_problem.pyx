from operator import itemgetter
from .observations import adapt_observation_representation
from .observations cimport calculate_sorted_rule_distances

cdef ConditionSide make_condition_side(condition_side, int feature, int feature_type, bool is_side_1):
    thresholds_ids = sorted(condition_side.items(), key=itemgetter(0))
    cdef cppvector[ValueIdsRanges] ids_values_c
    cdef ValueIdsRanges id_values_c
    cdef double threshold_c
    cdef int starts_at
    cdef int ends_at
    all_ids = []
    join = False

    bucket_thresholds = list(enumerate(thresholds_ids))

    for i in range(len(thresholds_ids)):
        bucket_id, (threshold, ids) = bucket_thresholds[i]

        if not join:
            starts_at = len(all_ids)

        all_ids.extend(sorted(ids))
        ends_at = len(all_ids)

        if feature_type == 2 and floor(threshold) == threshold:
            if i + 1 < len(bucket_thresholds):
                _, (next_threshold, _) = bucket_thresholds[i + 1]

                if floor(next_threshold) == threshold:
                    join = True
                    continue

            threshold += 0.5

        join = False
        id_values_c = ValueIdsRanges(
            threshold,
            starts_at,
            ends_at,
            ends_at - starts_at
        )
        ids_values_c.push_back(id_values_c)


    cdef cppvector[int] ids_c = all_ids

    return ConditionSide(
        ids_values_c,
        ids_c
    )


cdef ExtractionProblem make_problem(parsed_rf, dataset_description, bool search_closest_c, int log_every_c,
                                    int max_iterations_c):
    cdef cppvector[FeatureConditions] feature_conditions_list_c

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
        feature_conditions_list_c.push_back(feature_conditions_c)

    cdef cppvector[cppvector[double]] rule_probabilities_c = parsed_rf.rule_probs
    cdef cppvector[int] rule_tree_mapping_c = [parsed_rf.rule_tree[i] for i in range(len(parsed_rf.rule_tree))]
    cdef int n_features_c = len(dataset_description.dataset_description)
    cdef int n_trees_c = parsed_rf.n_trees
    cdef int n_labels_c = parsed_rf.n_labels
    cdef int n_rules_c = len(parsed_rf.rule_probs)

    cdef ExtractionProblem problem = ExtractionProblem(
        #problem.feature_conditions =
        feature_conditions_list_c,
        #problem.rule_probabilities =
        rule_probabilities_c,
        #problem.rule_tree =
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


cdef ConditionSide optimize_condition_side(ConditionSide & condition_side, to_remove):
    cdef ConditionSide optimized_condition_side
    cdef ValueIdsRanges current_ids_ranges

    cdef int i = 0
    cdef int start
    cdef int end
    cdef int rule_id

    for current_ids_ranges in condition_side.ids_values:
        start = optimized_condition_side.ids.size()

        for i in range(current_ids_ranges.ids_starts_at, current_ids_ranges.ids_ends_at):
            rule_id = condition_side.ids[i]
            if rule_id not in to_remove:
                optimized_condition_side.ids.push_back(rule_id)

        end = optimized_condition_side.ids.size()
        new_ids_values = ValueIdsRanges(current_ids_ranges.value, start, end, end - start)
        optimized_condition_side.ids_values.push_back(new_ids_values)

    return optimized_condition_side


cdef early_rule_pruning(ExtractionProblem & problem, cppvector[
        cpppair[int, double]] & sorted_rule_distances, double max_distance):
    cdef cpppair[int, double] current
    to_remove = set()

    for current in reversed(sorted_rule_distances):
        if current.second <= (max_distance + 0.001):
            break

        to_remove.add(current.first)

    return to_remove

cdef double estimate_min_distance(ExtractionProblem & problem, cppvector[cpppair[int, double]] & sorted_rule_distances, int factual_class):
    cdef cppvector[bool] set_trees = cppvector[bool](problem.n_trees, False)
    cdef int n_set_trees = 0
    cdef int i = 0
    cdef int t_id
    cdef cpppair[int, double] current
    cdef double distance
    cdef double acc_prob = 0
    cdef double target_prob = set_trees.size() / 2 # Assume problem is binary
    while True:
        current = sorted_rule_distances[i]
        rule_class = double_argmax(problem.rule_probabilities[current.first])
        t_id = problem.rule_tree[current.first]
        if rule_class != factual_class and not set_trees[t_id]:
            set_trees[t_id] = True
            n_set_trees += 1
            distance = current.second
            acc_prob += problem.rule_probabilities[current.first][rule_class]

        if n_set_trees == set_trees.size() or acc_prob > target_prob:
            break
        i+= 1

    return distance



cdef cppvector[FeatureConditions] optimize_feature_conditions_storage(ExtractionProblem & problem, to_remove):


    cdef cppvector[FeatureConditions] optimized_feature_conditions
    cdef FeatureConditions *current_feature_conditions
    cdef int i

    for i in range(problem.feature_conditions.size()):
        current_feature_conditions = &problem.feature_conditions[i]

        optimized_feature_conditions.push_back(FeatureConditions(
            current_feature_conditions.feature_type,
            optimize_condition_side(current_feature_conditions.side_1, to_remove),
            optimize_condition_side(current_feature_conditions.side_2, to_remove),
            current_feature_conditions.feat_range,
        ))

    return optimized_feature_conditions

cdef ExtractionContext create_extraction_state(ExtractionProblem & problem, parsed_rf, factual_class, observation,
                                               dataset_info, max_distance):
    cdef cppvector[bool] active_rules = cppvector[bool](problem.n_rules, True)
    cdef cppvector[FeatureBounds] feature_bounds
    cdef cppvector[cppvector[double]] tree_prob_sum
    cdef cppvector[int] active_by_tree = [0] * problem.n_trees
    cdef cppvector[SplitPoint] splits = cppvector[SplitPoint]()
    cdef double max_distance_c = max_distance
    cdef cppvector[bool] rule_backlist = cppvector[bool](problem.n_rules, False)
    cdef cppvector[double] current_obs = adapt_observation_representation(observation, dataset_info,
                                                                          include_all_categories=False)
    cdef cppvector[double] to_explain = adapt_observation_representation(observation, dataset_info,
                                                                         include_all_categories=False)

    cdef cppvector[cpppair[int, double]] sorted_rule_distances = calculate_sorted_rule_distances(problem, observation,
                                                                                                 parsed_rf,
                                                                                                 dataset_info)

    to_remove_rules = early_rule_pruning(problem, sorted_rule_distances, max_distance_c)
    cdef int sorted_rule_end_idx = problem.n_rules - len(to_remove_rules)
    cdef cppvector[FeatureConditions] feature_conditions_view = optimize_feature_conditions_storage(
        problem, to_remove_rules)

    cdef double min_possible_distance = estimate_min_distance(problem, sorted_rule_distances, factual_class)

    for _ in range(problem.n_trees):
        tree_prob_sum.push_back(cppvector[double](problem.n_labels, 0))

    cdef int num_active_rules = 0
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
        feature_bounds.push_back(FeatureBounds(cpppair[int, int](0, feat_conds.side_1.ids_values.size()), 0, False))

    return ExtractionContext(
        # Problem problem =
        problem,
        # cppvector[bool] active_rules  =
        active_rules,
        # cppvector[FeatureBounds] feature_bounds =
        feature_bounds,
        # cppvector[FeatureConditions] feature_conditions_view =
        feature_conditions_view,
        # cppvector[cppvector[double]] tree_prob_sum =
        tree_prob_sum,
        # cppvector[int] active_by_tree =
        active_by_tree,
        # cppvector[SplitPoint] splits =
        splits,
        # int num_active_rules =
        num_active_rules,
        # cppvector[double] current_obs =
        current_obs,
        #cppvector[double] to_explain =
        to_explain,
        #int real_class =
        factual_class,
        # cppvector[cpppair[int, double]] sorted_rule_distances =
        sorted_rule_distances,
        # int current_rule_filter,
        problem.n_rules - 1,
        # cppvector[bool] rule_blacklist,
        rule_backlist,
        # double max_distance =
        max_distance_c,
        # int sorted_rule_end_idx =
        sorted_rule_end_idx,
        # bool debug =
        False
    )

cdef cppvector[double] estimate_probability(ExtractionContext & global_state):
    cdef cppvector[double] current_prob = cppvector[double](global_state.problem.n_labels, 0)
    cdef double *tree_prob
    cdef int active_by_tree

    for i in range(global_state.problem.n_trees):
        active_by_tree = global_state.active_by_tree[i]
        tree_prob = &global_state.tree_prob_sum[i].at(0)

        for l in range(global_state.problem.n_labels):
            current_prob[l] += tree_prob[l] / (active_by_tree * global_state.problem.n_trees)

    return current_prob
