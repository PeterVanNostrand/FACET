import numpy as np
from operator import itemgetter

CATEGORY_ACTIVATED = 2
CATEGORY_DEACTIVATED = 1
CATEGORY_UNKNOWN = 3
CATEGORY_UNSET = 0
ACTIVATED_MASK = 0xAAAAAAAAAAAAAAAA


cdef double partial_distance(ExtractionProblem & problem, int feature, double to_explain, double counterfactual):
    cdef int num_features = problem.n_features
    cdef FeatureConditions *feature_conditions = &problem.feature_conditions[feature]
    cdef int feature_type = feature_conditions.feature_type
    cdef double feat_distance

    if feature_type == 4:
        if (ACTIVATED_MASK & (<long long> to_explain) & (<long long> counterfactual)) == 0:
            feat_distance = 1
    if feature_type == 5:
        return 0.0 if to_explain == counterfactual else 1.0
    else:
        feat_distance = fabs(to_explain - counterfactual) / feature_conditions.feat_range
        feat_distance = feat_distance

    return feat_distance / num_features

cdef double distance(ExtractionProblem & problem, cppvector[double] & to_explain, cppvector[double] & counterfactual):
    cdef FeatureConditions *feature_conditions
    cdef double distance = 0
    cdef int num_features = problem.n_features
    cdef int feature
    cdef int feature_type
    cdef double feat_distance
    for feature in range(num_features):
        distance += partial_distance(problem, feature, to_explain[feature], counterfactual[feature])
    return distance

cdef bool does_meet_split(ExtractionProblem & problem, cppvector[double] & counterfactual, SplitPoint & split_point):
    cdef int feature = split_point.feature
    cdef FeatureConditions *feature_conditions = &problem.feature_conditions[feature]
    cdef int feature_type = feature_conditions.feature_type
    cdef long long categorical_mask

    cdef int category
    cdef long long state

    if feature_type == 4:
        categorical_mask = <long long> counterfactual[split_point.feature]
        category = <int> split_point.value

        if split_point.meet:
            return is_category_state(categorical_mask, category, CATEGORY_ACTIVATED)
        else:
            return not is_category_state(categorical_mask, category, CATEGORY_ACTIVATED)
    else:
        return split_point.meet == (counterfactual[feature] <= split_point.value)

cdef bool update_to_meet_split(ExtractionProblem & problem, cppvector[double] & counterfactual,
                               SplitPoint & split_point, bool enable_if_last=False):
    cdef int feature = split_point.feature
    cdef FeatureConditions *feature_conditions = &problem.feature_conditions[feature]
    cdef int feature_type = feature_conditions.feature_type
    cdef bool is_mask_set = split_point.bounds_meet.is_mask_set if split_point.meet else split_point.bounds_not_meet.is_mask_set
    return update_to_meet(counterfactual, feature, feature_type, split_point.value, split_point.meet,
                          is_mask_set, problem.n_features, enable_if_last)

cdef bool update_to_meet(cppvector[double] & counterfactual, int feature, int feature_type,
                         double value, bool meet, bool is_mask_set=False, int num_categories=-1,
                         bool enable_if_last=False, double epsilon=0.0):
    cdef double offset = 0
    cdef int position
    cdef bool updated = False
    cdef long long category_mask

    if feature_type == 4 and value >= 20:
        print("Cannot work with categorical features with more than 20 levels")
        exit(0)

    if feature_type == 1:
        if (meet and value < counterfactual[feature]) or (not meet and value >= counterfactual[feature]):
            if feature_type == 1 and not meet:
                offset = epsilon

            counterfactual[feature] = value + offset
            updated = True
    elif feature_type == 2 and feature_type == 3:
        # print("UNREACHABLE CODE?")
        # print(str(feature_type))
        if (meet and value < counterfactual[feature]) or (not meet and value >= counterfactual[feature]):
            counterfactual[feature] = floor(value) if meet else floor(value + 1)
            updated = True
    elif feature_type == 4:
        position = <int> value
        category_mask = <long long> counterfactual[feature]
        if not meet:
            if enable_if_last and is_mask_set and num_categories != -1:
                for position in range(num_categories):
                    if is_category_state(category_mask, position, CATEGORY_UNSET):
                        counterfactual[feature] = <double> set_category_state(0, position, CATEGORY_ACTIVATED)
                        updated = True
                        break
            else:
                counterfactual[feature] = <double> set_category_state(category_mask, position,
                                                                      CATEGORY_DEACTIVATED)
                updated = True
        else:
            counterfactual[feature] = <double> set_category_state(0, position, CATEGORY_ACTIVATED)
            updated = True

    return updated

cdef bool is_category_state(long long category_mask, int category, int state) :
    if state == CATEGORY_UNSET:
        return (category_mask & (3 << 2 * category)) == 0

    cdef bool activated = (category_mask & (CATEGORY_ACTIVATED << 2 * category)) > 0
    cdef bool deactivated = (category_mask & (CATEGORY_DEACTIVATED << 2 * category)) > 0

    if state == CATEGORY_UNKNOWN:
        return activated and deactivated
    elif state == CATEGORY_DEACTIVATED:
        return deactivated and not activated
    else:
        return activated and not deactivated

cdef long long set_category_state(long long category_mask, int category, int state) :
    cdef long long clean_mask = ~(3 << 2 * category)
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

cdef cppvector[cpppair[int, double]] calculate_sorted_rule_distances(ExtractionProblem & problem, observation,
                                                                     parsed_rf, dataset_info):
    if LOG_LEVEL == "DEBUG":
        frule = open("logs/log_{}.txt".format(instance_num), "a")
        frule.write("######################### CALC SORTED RULE DISTANCE #########################\n")
        frule.write(ExtractionProblem_tostr(problem))
        frule.write("observation: " + str(observation) + "\n")
        frule.write("dataset_info:\n")
        frule.write(DatasetInfo_tostr(dataset_info))
        frule.write("parsed_rf:\n")
        frule.write(str(parsed_rf) + "\n")
        frule.close()

    rule_distances = []

    cdef cppvector[double] to_explain = adapt_observation_representation(observation, dataset_info,
                                                                         include_all_categories=False)
    if LOG_LEVEL == "DEBUG":
        frule = open("logs/log_{}.txt".format(instance_num), "a")
        frule.write("to_explain: " + str(to_explain) + "\n")
        frule.close()

    cdef cppvector[double] current_obs
    cdef double rule_dist
    for r_id, (t_id, rule) in enumerate(parsed_rf.rules):
        current_obs = adapt_observation_representation(observation, dataset_info, include_all_categories=True)

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

        rule_dist = distance(problem, to_explain, current_obs)
        rule_distances.append((r_id, rule_dist))
        
        if LOG_LEVEL == "DEBUG":
            frule = open("logs/log_{}.txt".format(instance_num), "a")
            frule.write("obs: " + str(current_obs) + "\n")
            frule.close()

    rule_distances = sorted(rule_distances, key=itemgetter(1))
    cdef cppvector[cpppair[int, double]] rule_distances_c = cppvector[cpppair[int, double]]()

    all_zero = True
    for r_id, r_distance in rule_distances:
        rule_distances_c.push_back(cpppair[int, double](r_id, r_distance))
        
        if LOG_LEVEL == "DEBUG":
            frule = open("logs/log_{}.txt".format(instance_num), "a")
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
