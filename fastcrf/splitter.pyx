from cython.parallel import parallel, prange
from cython.operator cimport dereference as deref, preincrement as inc
from .debug cimport LOG_LEVEL, ExtractionContext_tostr, FeatureConditions_tostr, ConditionSide_tostr, instance_num
from .debug_splitter cimport PartitionProb_tostr

cdef void agg_partition_prob(ExtractionContext & global_state, PartitionProb & partition_prob, int rule, bool add) :
    cdef int mult = 1 if add else -1
    cdef int tree_id = global_state.problem.rule_tree[rule]

    cdef double *tree_prob_sum = &partition_prob.tree_prob_sum[tree_id].at(0)
    cdef double *rule_prob = &global_state.problem.rule_probabilities[rule].at(0)

    cdef double prev_label_prob
    cdef double prob_delta_tree

    cdef int previous_active_by_tree = partition_prob.active_by_tree[tree_id]

    if LOG_LEVEL == "DEBUG":
        f = open("logs/log_{}.txt".format(instance_num), "a")
        f.write("mult (agg_partition_prob): " + str(mult) + "\n")
        f.write("tree_id (agg_partition_prob): " + str(tree_id) + "\n")
        f.write("previous_active_by_tree (agg_partition_prob): " + str(previous_active_by_tree) + "\n")
        f.write("global state in (agg_partition_prob) START\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global state in (agg_partition_prob) END\n")

    if partition_prob.active_by_tree[tree_id] == 1 and mult == -1:
        print("Attempting to remove last rule...", rule, "from tree", tree_id)
        exit(0)

    partition_prob.active_by_tree[tree_id] += mult
    partition_prob.num_active += mult

    if LOG_LEVEL == "DEBUG":
        f.write("partition_prob.active_by_tree[tree_id] (agg_partition_prob): " + str(partition_prob.active_by_tree[tree_id]) + "\n")
        f.write("partition_prob.num_active (agg_partition_prob): " + str(partition_prob.num_active) + "\n")

    for i in range(global_state.problem.n_labels):
        if previous_active_by_tree != 0:
            if LOG_LEVEL == "DEBUG":
                f.write("tree_prob_sum[i] (agg_partition_prob): " + str(tree_prob_sum[i]) + "\n")
            prev_label_prob = tree_prob_sum[i] / previous_active_by_tree
        else:
            prev_label_prob = 0
        tree_prob_sum[i] += mult * rule_prob[i]

        prob_delta_tree = (tree_prob_sum[i] / partition_prob.active_by_tree[tree_id]) - prev_label_prob
        partition_prob.prob[i] += prob_delta_tree / global_state.problem.n_trees

        if LOG_LEVEL == "DEBUG":
            f.write("prev_label_prob (agg_partition_prob): {:.4f}".format(prev_label_prob) + "\n")
            f.write("rule_prob[i] (agg_partition_prob): " + str(rule_prob[i]) + "\n")
            f.write("tree_prob_sum[i] (agg_partition_prob): " + str(tree_prob_sum[i]) + "\n")
            f.write("prob_delta_tree (agg_partition_prob): {:.4f}".format(prob_delta_tree) + "\n")
            f.write("partition_prob.prob[i] (agg_partition_prob): " + str(partition_prob.prob[i]) + "\n")
            f.write("global_state out START:" + "\n")
            f.write(ExtractionContext_tostr(global_state))
            f.write("global_state out END:" + "\n")
            f.write("partition_prob START:" + "\n")
            f.write(PartitionProb_tostr(partition_prob))
            f.write("partition_prob END:" + "\n")
            f.write("########### END agg_partition_prob ###########" + "\n")


cdef int batch_set_values_bucket_state(ExtractionContext & global_state, PartitionProb & partition, ConditionSide & condition_side,
                                 int bucket_from, int bucket_to, bool state):
    cdef int num_active = 0
    cdef int rule_id
    cdef int rule_pos
    cdef cppvector[int].iterator it = condition_side.ids.begin() + condition_side.ids_values[bucket_from].ids_starts_at
    cdef cppvector[int].iterator end = condition_side.ids.begin() + condition_side.ids_values[bucket_to].ids_ends_at

    if LOG_LEVEL == "DEBUG":
        f = open("logs/log_{}.txt".format(instance_num), "a")
        f.write("global_state (batch_set_values_bucket_state) START:\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global_state (batch_set_values_bucket_state) END:\n")
        f.write("partition (batch_set_values_bucket_state) START:\n")
        f.write(PartitionProb_tostr(partition))
        f.write("partition (batch_set_values_bucket_state) END:\n")
        f.write("ConditionSide (batch_set_values_bucket_state) START:\n")
        f.write(ConditionSide_tostr(condition_side))
        f.write("ConditionSide (batch_set_values_bucket_state) END:\n")


    if bucket_from <= bucket_to:
        while it != end:
            rule_id = deref(it)
            if LOG_LEVEL == "DEBUG":
                f.write("rule_id: " + str(rule_id) + "\n")
            inc(it)
            if global_state.active_rules[rule_id]:
                agg_partition_prob(global_state, partition, rule_id, state)
                num_active += 1
                if LOG_LEVEL == "DEBUG":
                    f.write("global_state (batch_set_values_bucket_state) START:\n")
                    f.write(ExtractionContext_tostr(global_state))
                    f.write("global_state (batch_set_values_bucket_state) END:\n")
                    f.write("partition (batch_set_values_bucket_state) START:\n")
                    f.write(PartitionProb_tostr(partition))
                    f.write("partition (batch_set_values_bucket_state) END:\n")

    return num_active

cdef int set_values_bucket_state(ExtractionContext & global_state, PartitionProb & partition, ConditionSide & condition_side,
                                 int bucket, bool state) :

    return batch_set_values_bucket_state(global_state, partition, condition_side, bucket, bucket, state)

cdef double compute_gain(PartitionProb & partition_meet, PartitionProb & partition_not_meet,
                         double global_score) :
    # Total number of rules. This does not represent the global number of rules, as there might be
    # rules that are in both partitions, so the total is likely higher than the global number of rules
    cdef int total = partition_not_meet.num_active + partition_meet.num_active
    cdef double meet_proportion = (<double> partition_meet.num_active) / total
    cdef double not_meet_proportion = 1 - meet_proportion
    cdef double result = global_score - (meet_proportion * criterion(partition_meet.prob) +
                           not_meet_proportion * criterion(partition_not_meet.prob))
    return result

cdef double calculate_category_score(ExtractionContext & global_state, FeatureConditions & conditions,
                                     PartitionProb partition_meet, PartitionProb partition_not_meet,
                                     double global_score, int category) :
    # Activate the 1 rules for the category (remember, currently all is set to 0s)
    cdef int num_activated = set_values_bucket_state(global_state, partition_meet, conditions.side_2, category, True)

    # Deactivate the 0 rules for the category
    set_values_bucket_state(global_state, partition_meet, conditions.side_1, category, False)

    # Deactivate the 1 rules for the category in the not meet partition, in this partition, both 1s and 0s rules
    # might be activated. Therefore, by deactivating the 1s rules, only the 0s rules remains
    set_values_bucket_state(global_state, partition_not_meet, conditions.side_2, category, False)
    return compute_gain(partition_meet, partition_not_meet, global_score)

cdef PartitionProb make_partition_prob(ExtractionContext & global_state, cppvector[double] & global_prob) :
    cdef PartitionProb partition_prob
    partition_prob.tree_prob_sum = cppvector[cppvector[double]](global_state.tree_prob_sum)
    partition_prob.active_by_tree = cppvector[int](global_state.active_by_tree)
    partition_prob.prob = cppvector[double](global_prob)
    partition_prob.num_active = global_state.num_active_rules
    return partition_prob

cdef SplitPointScore categorical_max_split(ExtractionContext & global_state, int feature, bool verbose) :
    cdef cppvector[double] global_prob = estimate_probability(global_state)
    cdef FeatureConditions * conditions = &global_state.feature_conditions_view[feature]
    cdef FeatureBounds * feature_bounds = &global_state.feature_bounds[feature]
    # assert conditions.feature_type == 4, "Categorical split is only valid for one-hot encoded variables"
    cdef int num_categories = conditions.side_1.ids_values.size()
    cdef long long categorical_mask = feature_bounds.categorical_mask

    # Partition that represents the current state
    cdef PartitionProb partition_not_meet = make_partition_prob(global_state, global_prob)

    # In this partition, all categories are negated (in one-hot, this would be all 0s, no categories activated)
    cdef PartitionProb partition_meet = make_partition_prob(global_state, global_prob)

    # Compute global score (needed to compute the uncertainty reduction)
    cdef double global_score = criterion(partition_meet.prob)

    # Setting all categories to 0s in the meet partition
    cdef int i = 0
    cdef int deactivated
    for i in range(num_categories):
        if is_category_state(categorical_mask, i, CATEGORY_UNSET):
            # Deactivate all rules that assert a column (disable all 1s in the one-hot)
            deactivated = set_values_bucket_state(global_state, partition_meet, conditions.side_2, i, False)
        elif is_category_state(categorical_mask, i, CATEGORY_ACTIVATED):
            print("Attempting to deactivate a category which was previously activated")
            exit(0)


    # Calculate split gain for each available category
    cdef int best_category = -1
    cdef double best_score = -1
    cdef double score
    cdef int num_active_categories = 0
    cdef bool is_split_ok = True

    for i in range(num_categories):
        if is_category_state(categorical_mask, i, CATEGORY_UNSET):
            score = calculate_category_score(global_state, global_state.feature_conditions_view[feature], partition_meet, partition_not_meet, global_score, i)
            num_active_categories += 1

            if best_score == -1 or score > best_score:
                best_score = score
                best_category = i

    # Bounds for the split
    cdef long long new_categorical_mask_meet = 0x5555555555555555 # All disabled
    cdef long long new_categorical_mask_not_meet = categorical_mask

    # Activate category in meet split, and disable in not met split
    new_categorical_mask_meet = set_category_state(new_categorical_mask_meet, best_category, CATEGORY_ACTIVATED)
    new_categorical_mask_not_meet = set_category_state(new_categorical_mask_not_meet, best_category, CATEGORY_DEACTIVATED)

    # If there is only two categories activated, then the remaining category should be activated in the not
    # meet side
    if num_active_categories == 2:
        new_categorical_mask_not_meet = 0x5555555555555555 # All disabled
        for i in range(num_categories):
            if is_category_state(categorical_mask, i, CATEGORY_UNSET) and i != best_category:
                new_categorical_mask_not_meet = set_category_state(new_categorical_mask_not_meet, i, CATEGORY_ACTIVATED)
                break

    elif num_active_categories == 0:
        feature_bounds.is_mask_set = True
        is_split_ok = False


    # Create the new bounds
    cdef cpppair[int, int] zero_pair = cpppair[int, int](0, 0)
    cdef FeatureBounds new_bounds_meet = make_feature_bounds(zero_pair, new_categorical_mask_meet, True)
    cdef FeatureBounds new_bounds_not_meet = make_feature_bounds(zero_pair, new_categorical_mask_not_meet, num_active_categories == 2)
    cdef SplitPoint split_point = make_split_point(best_category, feature, True, new_bounds_meet, new_bounds_not_meet)
    return make_split_point_score(best_score, split_point, is_split_ok)


cdef SplitPointScore make_split_point_score(double score, SplitPoint & split_point, bool is_ok) :
    cdef SplitPointScore split_score
    split_score.score = score
    split_score.split_point = split_point
    split_score.is_ok = is_ok
    return split_score

cdef SplitPoint make_split_point(double value, int feature, bool meet, FeatureBounds & bmeet, FeatureBounds & bnot_meet) :
    cdef SplitPoint split_point
    split_point.value = value
    split_point.feature = feature
    split_point.meet = meet
    split_point.bounds_meet = bmeet
    split_point.bounds_not_meet = bnot_meet
    return split_point

cdef FeatureBounds make_feature_bounds(cpppair[int, int] bounds, long categorical_mask, bool is_mask_set) :
    cdef FeatureBounds feature_bounds
    feature_bounds.numerical_bounds = bounds
    feature_bounds.categorical_mask = categorical_mask
    feature_bounds.is_mask_set = is_mask_set
    return feature_bounds

cdef SplitPointScore numerical_max_split(ExtractionContext & global_state, int feature):
    if LOG_LEVEL == "DEBUG":
        f = open("logs/log_{}.txt".format(instance_num), "a")
        f.write("NUM MAX SPLIT 1\n")
        f.write("global_state (numerical_max_split):\n")
        f.write(ExtractionContext_tostr(global_state))
    
    cdef cppvector[double] global_prob = estimate_probability(global_state)
    cdef FeatureConditions * conditions = &global_state.feature_conditions_view[feature]
    cdef FeatureBounds * feature_bounds = &global_state.feature_bounds[feature]
    if LOG_LEVEL == "DEBUG":
        f.write("NUM MAX SPLIT 2\n")
        f.write("global_state (numerical_max_split):\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global_prob (numerical_max_split): " + str(global_prob) + "\n")

    # Partition for rules that meet feat > x
    cdef PartitionProb partition_not_meet = make_partition_prob(global_state, global_prob)
    if LOG_LEVEL == "DEBUG":
        f.write("NUM MAX SPLIT 3\n")
        f.write("global_state (numerical_max_split):\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global_prob (numerical_max_split): " + str(global_prob) + "\n")
        f.write("partition_not_meet (numerical_max_split):\n")
        f.write(PartitionProb_tostr(partition_not_meet))

    # Partition for rules that meet feat <= x
    cdef PartitionProb partition_meet = make_partition_prob(global_state, global_prob)
    if LOG_LEVEL == "DEBUG":
        f.write("NUM MAX SPLIT 4\n")
        f.write("global_state (numerical_max_split):\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global_prob (numerical_max_split): " + str(global_prob) + "\n")
        f.write("partition_not_meet (numerical_max_split):\n")
        f.write(PartitionProb_tostr(partition_not_meet))
        f.write("partition_meet (numerical_max_split):\n")
        f.write(PartitionProb_tostr(partition_meet))

    # Compute global score (needed to compute the uncertainty reduction)
    cdef double global_score = criterion(partition_meet.prob)
    if LOG_LEVEL == "DEBUG":
        f.write("NUM MAX SPLIT 5\n")
        f.write("global_state (numerical_max_split):\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global_prob (numerical_max_split): " + str(global_prob) + "\n")
        f.write("partition_not_meet (numerical_max_split):\n")
        f.write(PartitionProb_tostr(partition_not_meet))
        f.write("partition_meet (numerical_max_split):\n")
        f.write(PartitionProb_tostr(partition_meet))
        f.write("global_score: {:.4f}".format(global_score) + "\n")


    cdef int min_bucket = feature_bounds.numerical_bounds.first
    cdef int max_bucket = feature_bounds.numerical_bounds.second
    cdef int current_bucket = min_bucket

    # Deactivate greater than conditions (side_2) in the meet partition, because at the beginning,
    # the partition starts with the rules which meet the condition feat <= min(x), thus, all greater than
    # conditions are deactivated
    for bucket in range(min_bucket, max_bucket):
        set_values_bucket_state(global_state, partition_meet, conditions.side_2, bucket, False)

    if LOG_LEVEL == "DEBUG":
        f.write("NUM MAX SPLIT 6\n")
        f.write("global_state (numerical_max_split):\n")
        f.write(ExtractionContext_tostr(global_state))
        f.write("global_prob (numerical_max_split): " + str(global_prob) + "\n")
        f.write("partition_not_meet (numerical_max_split):\n")
        f.write(PartitionProb_tostr(partition_not_meet))
        f.write("partition_meet (numerical_max_split):\n")
        f.write(PartitionProb_tostr(partition_meet))
        f.write("global_score: {:.4f}".format(global_score) + "\n")
        f.write("min_bucket: " + str(min_bucket) + "\n")
        f.write("max_bucket: " + str(max_bucket) + "\n")
        f.write("current_bucket: " + str(current_bucket) + "\n")
        # f.write("current_score: " + str(current_score) + "\n")
        # f.write("best_threshold: " + str(best_threshold) + "\n")
        # f.write("best_score: " + str(best_score) + "\n")
        # f.write("best_position: " + str(best_position) + "\n")
        f.close()

    # In this iteration, the partition meet holds all lte rules and does not have any gt
    # since the threshold is the smallest (rules defined in <= smallest).
    # In the not meet partition, we deactivate the rules whose threshold is == smallest, since this partition
    # contains all rules defined in > threshold (and <= threshold is not)
    set_values_bucket_state(global_state, partition_not_meet, conditions.side_1, current_bucket, False)

    cdef double current_score = compute_gain(partition_meet, partition_not_meet, global_score)
    cdef double best_threshold = conditions.side_1.ids_values[current_bucket].value
    cdef double best_score = current_score
    cdef int best_position = current_bucket
    current_bucket += 1

    cdef int num_changed

    while current_bucket < max_bucket:
        # Notice that, a bucket i refers to the same threshold in side_1 and side_2
        # for simplicity, we refer to these number as threshold in the next comments

        # meet partition: all rules that satisfy rule <= threshold (all lte rules, gt rules that threshold(rule) < threshold
        # not meet partition: all rules that satisfy rule > threshold (all gt rules, lte rules that threshold(rule) > threshold)
        num_changed =  set_values_bucket_state(global_state, partition_meet, conditions.side_2, current_bucket - 1, True)

        # Deactivating lte rules with rule(threshold) < threshold, because there are no longer greater than threshold
        # since this partition know only holds rules that meet rule > threshold
        num_changed += set_values_bucket_state(global_state, partition_not_meet, conditions.side_1, current_bucket, False)

        if num_changed > 0:
            # Compute the current gain for the split
            current_gain = compute_gain(partition_meet, partition_not_meet, global_score)

            if current_gain > best_score:
                best_score = current_gain
                best_threshold = conditions.side_1.ids_values[current_bucket].value
                best_position = current_bucket

        current_bucket += 1

    cdef FeatureBounds new_bounds_meet = make_feature_bounds(cpppair[int, int](min_bucket, best_position), 0, False)
    cdef FeatureBounds new_bounds_not_meet = make_feature_bounds(cpppair[int, int](best_position + 1, max_bucket), 0, False)
    cdef SplitPoint split_point = make_split_point(best_threshold, feature, True, new_bounds_meet, new_bounds_not_meet)
    return make_split_point_score(best_score, split_point, True)

cdef SplitPointScore calculate_feature_split(ExtractionContext & global_state, int feature, bool verbose) :
    cdef FeatureConditions * feature_conditions = &global_state.feature_conditions_view[feature]
    if LOG_LEVEL == "DEBUG":
        f = open("logs/log_{}.txt".format(instance_num), "a")
        f.write("feature_conditions (calculate_feature_split):\n")
        f.write(FeatureConditions_tostr(global_state.feature_conditions_view[feature]))
        f.close()
    if feature_conditions.feature_type == 4:
        # One-hot
        return categorical_max_split(global_state, feature, verbose)
    else:
        # Real, ordinal and binary
        return numerical_max_split(global_state, feature)

cdef cpppair[bool, SplitPoint] calculate_split(ExtractionContext & global_state, bool verbose=False):
    cdef SplitPointScore current_split
    cdef FeatureBounds * feature_bounds
    cdef int feature_type

    cdef FeatureBounds mock_feature_bounds = make_feature_bounds(cpppair[int, int](0, 0), 0, False)
    cdef SplitPoint mock_split_point = make_split_point(0, -1, False, mock_feature_bounds, mock_feature_bounds)
    cdef SplitPointScore best_split = make_split_point_score(-1, mock_split_point, False)

    cdef size_t n_features = global_state.problem.n_features
    cdef int feature

    for feature in range(n_features):
        feature_bounds = &global_state.feature_bounds[feature]
        feature_type = global_state.feature_conditions_view[feature].feature_type

        if (feature_type == 4 and not feature_bounds.is_mask_set) or \
                (feature_type != 4 and feature_bounds.numerical_bounds.first < feature_bounds.numerical_bounds.second):
            if LOG_LEVEL == "DEBUG":
                f = open("logs/log_{}.txt".format(instance_num), "a")
                f.write("global_state (calculate_split):\n")
                f.write(ExtractionContext_tostr(global_state))
                f.write("feature: " + str(feature) + "\n")
                f.write("verbose: " + str(verbose) + "\n")
                f.close()
            current_split = calculate_feature_split(global_state,  feature, verbose)

            if not best_split.is_ok or (current_split.is_ok and current_split.score > best_split.score):
                best_split = current_split

    return cpppair[bool, SplitPoint](best_split.is_ok, best_split.split_point)

# cdef double gini(cppvector[double] & label_probs) :
#         cdef double score = 0
#         for prob in label_probs:
#             score += prob * prob
#         return 1 - score

# Shannon entropy
cdef double criterion(cppvector[double] & label_probs) :
    cdef double score = 0
    for prob in label_probs:
        if prob > 0:
            score +=  prob * log2(prob)
    return -score
