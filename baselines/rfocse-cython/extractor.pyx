from cython.operator import dereference
from .debug cimport LOG_LEVEL, Solution_tostr, ExtractionContext_tostr, instance_num

cdef void debug(ExtractionContext & global_state):
    print("Num activate rules", global_state.num_active_rules, "Estimated probability ", estimate_probability(global_state))
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


cdef bool activate_rule(ExtractionContext & global_state, int rule_id) :
    if global_state.active_rules[rule_id]:
        raise ValueError()

    if global_state.rule_blacklist[rule_id]:
        # This rule has been previously filtered
        return False

    cdef int tree_id = global_state.problem.rule_tree[rule_id]

    global_state.active_rules[rule_id] = True
    global_state.num_active_rules += 1

    global_state.active_by_tree[tree_id] += 1
    sum_vector(global_state.problem.rule_probabilities[rule_id], global_state.tree_prob_sum[tree_id])
    return True

cdef void deactivate_rule(ExtractionContext & global_state, int rule_id) :
    if not global_state.active_rules[rule_id]:
        raise ValueError()

    cdef int tree_id = global_state.problem.rule_tree[rule_id]

    if global_state.active_by_tree[tree_id] == 1:
        print("Trying to remove last rule from tree", tree_id, "rule", rule_id)
        debug(global_state)
        exit(0)

    global_state.active_rules[rule_id] = False
    global_state.num_active_rules -= 1

    global_state.active_by_tree[tree_id] -= 1
    sub_vector(global_state.problem.rule_probabilities[rule_id], global_state.tree_prob_sum[tree_id])


cdef int early_stop(ExtractionContext & global_state):
    cdef int num_set = 0
    cdef cppvector[double] set_probs = cppvector[double](global_state.problem.n_labels, 0)
    cdef int t_id = 0
    cdef int l = 0
    cdef int non_zero = -1

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


    cdef int max_label = double_argmax(set_probs)
    cdef int num_unset = global_state.problem.n_trees - num_set

    cdef double worst_case_prob = 1.0 * num_unset

    for l in range(global_state.problem.n_labels):
        if l != max_label and set_probs[max_label] <= (set_probs[l] + worst_case_prob):
            return -1

    return max_label

cdef int can_stop(ExtractionContext & global_state):
    cdef cppvector[double] labels_probs
    cdef cppvector[double] rule_prob

    cdef int estimated_class = -1

    if global_state.num_active_rules == global_state.problem.n_trees:
        labels_probs = cppvector[double](global_state.problem.n_labels, 0.0)

        for i in range(global_state.active_rules.size()):
            if global_state.active_rules[i]:
                rule_prob = global_state.problem.rule_probabilities[i]

                for l in range(global_state.problem.n_labels):
                    labels_probs[l] += rule_prob[l]

        estimated_class = double_argmax(labels_probs)
    else:
        estimated_class = early_stop(global_state)

    return estimated_class


cdef bool is_bucket_active(ExtractionContext & global_state, ConditionSide & condition_side, int bucket):
    for rule_pos in range(condition_side.ids_values[bucket].ids_starts_at,
                              condition_side.ids_values[bucket].ids_ends_at):
        rule_id = condition_side.ids[rule_pos]
        if global_state.active_rules[rule_id] and not global_state.rule_blacklist[rule_id]:
            return True
    return False


cdef void update_bounds_categorical(ExtractionContext & global_state, int feature):
    cdef FeatureBounds * feature_bounds = &global_state.feature_bounds[feature]
    cdef FeatureConditions * feature_conditions = &global_state.feature_conditions_view[feature]
    cdef int num_categories
    num_categories = feature_conditions.side_1.ids_values.size()
    for bucket in range(num_categories):
        if is_category_state(feature_bounds.categorical_mask, bucket, CATEGORY_UNSET) and \
                not (is_bucket_active(global_state, feature_conditions.side_1, bucket) and
                    is_bucket_active(global_state, feature_conditions.side_2, bucket)):

            feature_bounds.categorical_mask = set_category_state(feature_bounds.categorical_mask,
                                                                 bucket,
                                                                 CATEGORY_UNKNOWN)


cdef void soft_numerical_pruning(ExtractionContext & global_state, int feature, Solution & solution, cppvector[int] & removed_rules, bool deactivate=True, bool feature_only_dist=False):
    cdef FeatureBounds * feature_bounds = &global_state.feature_bounds[feature]
    cdef FeatureConditions * feature_conditions = &global_state.feature_conditions_view[feature]
    cdef ConditionSide * side_1 = &feature_conditions.side_1
    cdef ConditionSide * side_2 = &feature_conditions.side_2
    cdef int min_bucket = feature_bounds.numerical_bounds.first
    cdef int max_bucket = feature_bounds.numerical_bounds.second
    cdef double current_solution_distance = solution.distance if solution.found else global_state.max_distance
    cdef int i
    cdef double factual_value = global_state.to_explain[feature]
    cdef double current_value = global_state.current_obs[feature]

    max_bucket -= 1
    cdef double updated_value

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
            deactivate_bucket(global_state, dereference(side_1), min_bucket, removed_rules)

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
            deactivate_bucket(global_state, dereference(side_2), max_bucket, removed_rules)
        max_bucket -= 1


    global_state.current_obs[feature] = current_value

    if max_bucket < min_bucket:
        feature_bounds.numerical_bounds.first = min_bucket
        feature_bounds.numerical_bounds.second = min_bucket
    else:
        feature_bounds.numerical_bounds.first = min_bucket
        feature_bounds.numerical_bounds.second = max_bucket + 1


cdef void update_bounds_numerical(ExtractionContext & global_state, int feature):
    cdef FeatureBounds * feature_bounds = &global_state.feature_bounds[feature]
    cdef FeatureConditions * feature_conditions = &global_state.feature_conditions_view[feature]
    cdef ConditionSide * side_1 = &feature_conditions.side_1
    cdef ConditionSide * side_2 = &feature_conditions.side_2
    cdef int min_bucket = feature_bounds.numerical_bounds.first
    cdef int max_bucket = feature_bounds.numerical_bounds.second
    max_bucket -= 1

    while min_bucket <= max_bucket and not is_bucket_active(global_state, feature_conditions.side_1, min_bucket):
        min_bucket += 1

    while min_bucket <= max_bucket and not is_bucket_active(global_state, feature_conditions.side_2, max_bucket):
        max_bucket -= 1

    if max_bucket < min_bucket:
        feature_bounds.numerical_bounds.first = min_bucket
        feature_bounds.numerical_bounds.second = min_bucket
    else:
        feature_bounds.numerical_bounds.first = min_bucket
        feature_bounds.numerical_bounds.second = max_bucket + 1

cdef void soft_pruning(ExtractionContext & global_state, Solution & solution, cppvector[int] & removed_rules):
    cdef FeatureConditions * feature_conditions

    for feature in range(global_state.problem.n_features):
        feature_conditions = &global_state.feature_conditions_view[feature]
        if feature_conditions.feature_type != 4:
            soft_numerical_pruning(global_state, feature, solution, removed_rules)

cdef void ensure_bounds_consistency(ExtractionContext & global_state, Solution & solution):
    cdef int feature = 0
    cdef FeatureConditions * feature_conditions = &global_state.feature_conditions_view[feature]
    cdef cppvector[int] removed

    for feature in range(global_state.problem.n_features):
        if feature_conditions.feature_type == 4:
            update_bounds_categorical(global_state, feature)
        else:
            update_bounds_numerical(global_state, feature)
            assert removed.size() == 0

cdef void deactivate_bucket_batch(ExtractionContext & global_state, ConditionSide & condition_side, int bucket_from, int bucket_to, cppvector[int] & removed_rules):
    cdef int rule_id
    cdef int rule_pos
    if bucket_from <= bucket_to:
        for rule_pos in range(condition_side.ids_values[bucket_from].ids_starts_at,
                              condition_side.ids_values[bucket_to].ids_ends_at):
            rule_id = condition_side.ids[rule_pos]

            if global_state.active_rules[rule_id]:
                deactivate_rule(global_state, rule_id)
                removed_rules.push_back(rule_id)

cdef void deactivate_bucket(ExtractionContext & global_state, ConditionSide & condition_side, int bucket, cppvector[int] & removed_rules):
    deactivate_bucket_batch(global_state, condition_side, bucket, bucket, removed_rules)

cdef StepState apply_feature_bounds_categorical(ExtractionContext & global_state, SplitPoint & split_point):
    cdef cppvector[int] removed_rules
    cdef FeatureBounds * previous_feature_bounds = &global_state.feature_bounds[split_point.feature]
    cdef FeatureConditions * feature_conditions = &global_state.feature_conditions_view[split_point.feature]
    cdef int num_categories = feature_conditions.side_1.ids_values.size()

    cdef FeatureBounds feature_bounds = split_point.bounds_meet if split_point.meet else split_point.bounds_not_meet

    cdef long long previous_mask = previous_feature_bounds.categorical_mask
    cdef long long current_mask = feature_bounds.categorical_mask

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

    cdef StepState state = StepState(removed_rules, cppvector[FeatureBounds](global_state.feature_bounds))
    global_state.feature_bounds[split_point.feature] = feature_bounds
    return state


cdef StepState apply_feature_bounds_numerical(ExtractionContext & global_state, SplitPoint & split_point):
    cdef cppvector[int] removed_rules
    cdef FeatureBounds * previous_feature_bounds = &global_state.feature_bounds[split_point.feature]
    cdef FeatureConditions * feature_conditions = &global_state.feature_conditions_view[split_point.feature]

    cdef int previous_min_bucket = previous_feature_bounds.numerical_bounds.first
    cdef int previous_max_bucket = previous_feature_bounds.numerical_bounds.second

    cdef FeatureBounds feature_bounds = split_point.bounds_meet if split_point.meet else split_point.bounds_not_meet
    cdef FeatureBounds feature_bounds_not = split_point.bounds_meet if not split_point.meet else split_point.bounds_not_meet

    cdef int current_min_bucket = feature_bounds.numerical_bounds.first
    cdef int current_max_bucket = feature_bounds.numerical_bounds.second


    deactivate_bucket_batch(global_state, feature_conditions.side_2, current_max_bucket, previous_max_bucket - 1, removed_rules)
    deactivate_bucket_batch(global_state, feature_conditions.side_1, previous_min_bucket, current_min_bucket - 1, removed_rules)

    cdef StepState state = StepState(removed_rules, cppvector[FeatureBounds](global_state.feature_bounds))
    global_state.feature_bounds[split_point.feature] = feature_bounds
    return state

cdef StepState apply_feature_bounds(ExtractionContext & global_state, Solution & solution, SplitPoint & split_point):
    cdef FeatureConditions * feature_conditions = &global_state.feature_conditions_view[split_point.feature]
    cdef StepState state

    if feature_conditions.feature_type == 4:
        state = apply_feature_bounds_categorical(global_state, split_point)
    else:
        state = apply_feature_bounds_numerical(global_state, split_point)

    global_state.splits.push_back(split_point)
    soft_pruning(global_state, solution, state.removed_rules)
    ensure_bounds_consistency(global_state, solution)
    return state


cdef void rollback_split(ExtractionContext & global_state, StepState & step_state):
    global_state.feature_bounds = step_state.previous_bounds
    global_state.splits.pop_back()
    for rule_id in step_state.removed_rules:
        activate_rule(global_state, rule_id)

cdef cppvector[int] get_set_trees_ids(ExtractionContext & global_state):
    cdef cppvector[int] set_rules = cppvector[int]()
    cdef int t_id
    for r_id in range(global_state.problem.n_rules):
        t_id = global_state.problem.rule_tree[r_id]
        if global_state.active_by_tree[t_id] == 1 and global_state.active_rules[r_id]:
            set_rules.push_back(r_id)
    return set_rules


cdef cppvector[int] get_set_rules_ids(ExtractionContext & global_state):
    cdef cppvector[int] set_rules = cppvector[int]()
    cdef int t_id
    for r_id in range(global_state.problem.n_rules):
        if global_state.active_rules[r_id]:
            set_rules.push_back(r_id)
    return set_rules

cdef double calculate_current_distance(ExtractionContext & global_state):
    return rule_distance(global_state.problem, global_state.to_explain,
                                             global_state.current_obs)


cdef bool prune_rules(ExtractionContext & global_state, Solution & solution):
    cdef double current_solution_distance = solution.distance if solution.found else global_state.max_distance

    if current_solution_distance < 0:
        # A solution has not been found yet
        return False

    cdef int current_rule_pruning = global_state.current_rule_pruning
    cdef int rule_id
    while current_rule_pruning > 0 and \
            global_state.sorted_rule_distances[current_rule_pruning].second > current_solution_distance:

        rule_id = global_state.sorted_rule_distances[current_rule_pruning].first
        t_id = global_state.problem.rule_tree[rule_id]
        global_state.rule_blacklist[rule_id] = True

        if global_state.active_rules[rule_id]:
            deactivate_rule(global_state, rule_id)

        current_rule_pruning -= 1

    cdef bool changed = global_state.current_rule_pruning != current_rule_pruning
    global_state.current_rule_pruning = current_rule_pruning
    return changed


cdef void extract_counterfactual_impl(ExtractionContext & global_state, Solution & solution):
    solution.num_evaluations += 1
    cdef int iteration = solution.num_evaluations

    if LOG_LEVEL == "DEBUG":
        f = open("logs/log_{}.txt".format(instance_num), "a")
        f.write("global_state (extract_counterfactual_impl) in START:\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global_state (extract_counterfactual_impl) in END:\n")
        f.write("solution (extract_counterfactual_impl) in START:\n")
        f.write(Solution_tostr(solution))
        f.write("solution (extract_counterfactual_impl) in END:\n")
        f.write("CHECKING CONDITIONS\n")

    if (solution.found and not global_state.problem.search_closest) \
            or (global_state.problem.max_iterations != -1 and iteration >= global_state.problem.max_iterations):
        return

    if LOG_LEVEL == "DEBUG":
        f.write("solution:\n")
        f.write(Solution_tostr(solution))
        f.write("PRUNING RULES\n")

    if prune_rules(global_state, solution) or solution.num_evaluations == 1:
        ensure_bounds_consistency(global_state, solution)

    if LOG_LEVEL == "DEBUG":
        f.write("solution:\n")
        f.write(Solution_tostr(solution))

    cdef int estimated_class = can_stop(global_state)

    cdef StepState step_state
    cdef double previous_value

    cdef SplitPoint split_point
    cdef cpppair[bool, SplitPoint] split_point_ok

    if iteration > 0 and global_state.problem.log_every > 0 and (iteration % global_state.problem.log_every) == 0:
        printf("Iteration number [%d]. Discovered foil [%d]. Current solution distance [%.06f]. Current distance [%.06f]. Pruned rules [%d] out of [%d]. Max distance [%.06f]\n",
               iteration,
               solution.num_discovered_non_foil,
               solution.distance,
               calculate_current_distance(global_state),
               global_state.problem.n_rules - global_state.current_rule_pruning - 1,
               global_state.problem.n_rules,
               global_state.max_distance
               )

    if LOG_LEVEL == "DEBUG":
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
            solution.conditions = cppvector[SplitPoint](global_state.splits)
            solution.set_rules = get_set_rules_ids(global_state)
            solution.distance = calculate_current_distance(global_state)
            solution.instance = cppvector[double](global_state.current_obs)
            solution.estimated_class = estimated_class
    else:
        verbose = False
        split_point_ok = calculate_split(global_state, verbose)

        if not split_point_ok.first:
            debug(global_state)
            exit(0)

        max_distance = solution.distance if solution.found else global_state.max_distance
        split_point = split_point_ok.second
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
