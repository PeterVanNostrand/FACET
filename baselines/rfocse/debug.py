from .extraction_problem_header import FeatureConditions, ValueIdsRanges, ExtractionProblem, FeatureBounds, SplitPoint, ExtractionContext, Solution, ConditionSide
from .datasets import DatasetInfo
import json

LOG_LEVEL = "NONE"
global instance_num
instance_num = 0


def ConditionSide_tostr(condition_side: ConditionSide) -> str:
    # cdef struct ConditionSide:
    #     cppvector[ValueIdsRanges] ids_values
    #     cppvector[int] ids
    out_str = ""
    out_str += "\t\tids: " + numerical_list_tostr(condition_side.ids) + "\n"
    for vidr in condition_side.ids_values:
        out_str += "\t\tvidr[val: {}, idstart: {}, idend: {}, size: {}]\n".format(
            vidr.value, vidr.ids_starts_at, vidr.ids_ends_at, vidr.size)
    return out_str


def FeatureConditions_tostr(myconditions: FeatureConditions):
    # cdef struct FeatureConditions:
    #     int feature_type
    #     ConditionSide side_1
    #     ConditionSide side_2
    #     double feat_range
    out_str = "\ttype: {}\n".format(myconditions.feature_type)  # conditions.feature_type
    out_str += "\tside 1:\n"
    out_str += ConditionSide_tostr(myconditions.side_1)
    out_str += "\tside 2:\n"
    out_str += ConditionSide_tostr(myconditions.side_2)
    return out_str


def ExtractionProblem_tostr(problem: ExtractionProblem):
    out_str = "Extraction Problem:\n"
    out_str += "feature_conditions\n"

    for i in range(len(problem.feature_conditions)):
        conditions: FeatureConditions = problem.feature_conditions[i]
        out_str += FeatureConditions_tostr(conditions)
    out_str += "rule_probabilities:\n"
    for probs in problem.rule_probabilities:
        out_str += "\t[{}, {}]\n".format(probs[0], probs[1])
    out_str += "\n"
    out_str += "rule_tree: " + str(problem.rule_tree) + "\n"
    out_str += "n_rules: " + str(problem.n_rules) + "\n"
    out_str += "n_features: " + str(problem.n_features) + "\n"
    out_str += "n_labels: " + str(problem.n_labels) + "\n"
    out_str += "n_trees: " + str(problem.n_trees) + "\n"
    out_str += "search_closest: " + str(problem.search_closest) + "\n"
    out_str += "log_every: " + str(problem.log_every) + "\n"
    out_str += "max_iterations: " + str(problem.max_iterations) + "\n"

    return out_str


def FeatureBounds_tostr(bounds: FeatureBounds):
    # cdef struct FeatureBounds:
    #     cpppair[int, int] numerical_bounds
    #     long long categorical_mask
    #     bool is_mask_set
    out_str = "\tbound:\n"
    out_str += "\t\tnumerical_bounds: ({}, {})\n".format(bounds.numerical_bounds[0], bounds.numerical_bounds[1])
    out_str += "\t\tmask: " + str(bounds.categorical_mask) + "\n"
    out_str += "\t\tismask: " + str(bounds.is_mask_set) + "\n"
    return out_str


def SplitPoint_tostr(split: SplitPoint):
    # cdef struct SplitPoint:
    #     double value
    #     int feature
    #     bool meet
    #     FeatureBounds bounds_meet
    #     FeatureBounds bounds_not_meet
    out_str = ""
    out_str += "\tvalue: " + str(split.value) + "\n"
    out_str += "\tfeature: " + str(split.feature) + "\n"
    out_str += "\tmeet: " + str(split.meet) + "\n"
    out_str += "\tbounds_meet:\n"
    out_str += FeatureBounds_tostr(split.bounds_meet)
    out_str += "\tbounds_not_meet:\n"
    out_str += FeatureBounds_tostr(split.bounds_not_meet)
    return out_str


def ExtractionContext_tostr(global_state: ExtractionContext):
    # cdef struct ExtractionProblem:
    #     cppvector[bool] active_rules
    #     cppvector[FeatureBounds] feature_bounds
    #     cppvector[FeatureConditions] feature_conditions_view
    #     cppvector[cppvector[double]] tree_prob_sum
    #     cppvector[int] active_by_tree
    #     cppvector[SplitPoint] splits
    #     int num_active_rules
    #     cppvector[double] current_obs
    #     cppvector[double] to_explain
    #     int real_class
    #     cppvector[cpppair[int, double]] sorted_rule_distances
    #     int current_rule_pruning
    #     cppvector[bool] rule_blacklist
    #     double max_distance
    #     int sorted_rule_end_idx
    #     bool debug
    out_str = "Extraction Context:\n"
    out_str += ExtractionProblem_tostr(global_state.problem)
    out_str += "active_rules: " + str(global_state.active_rules) + "\n"
    out_str += "feature_bounds:\n"
    for bound in global_state.feature_bounds:
        out_str += FeatureBounds_tostr(bound)
    out_str += "feature_conditions_view\n"
    for cond in global_state.feature_conditions_view:
        out_str += FeatureConditions_tostr(cond)
    out_str += "tree_prob_sum:\n"
    for probs in global_state.tree_prob_sum:
        out_str += "\t" + str(probs) + "\n"
    out_str += "active_by_tree: " + str(global_state.active_by_tree) + "\n"
    out_str += "splits:\n"
    for split in global_state.splits:
        out_str += SplitPoint_tostr(split)
    out_str += "num_active_rules: " + str(global_state.num_active_rules) + "\n"
    out_str += "current_obs: ["
    for i in range(len(global_state.current_obs)):
        out_str += str(global_state.current_obs[i])
        if i != len(global_state.current_obs)-1:
            out_str += ", "
    out_str += "]\n"
    out_str += "to_explain: ["
    for i in range(len(global_state.to_explain)):
        out_str += str(global_state.to_explain[i])
        if i != len(global_state.to_explain)-1:
            out_str += ", "
    out_str += "]\n"
    out_str += "real_class: " + str(global_state.real_class) + "\n"
    out_str += "sorted_rule_distances:\n"
    for dists in global_state.sorted_rule_distances:
        out_str += "\t" + str(dists) + "\n"
    out_str += "current_rule_pruning: " + str(global_state.current_rule_pruning) + "\n"
    out_str += "rule_blacklist: " + str(global_state.rule_blacklist) + "\n"
    out_str += "max_distance: " + str(global_state.max_distance) + "\n"
    out_str += "sorted_rule_end_idx: " + str(global_state.sorted_rule_end_idx) + "\n"
    out_str += "debug: " + str(global_state.debug) + "\n"
    return out_str


def DatasetInfo_tostr(dataset_info: DatasetInfo):
    out_str = ""
    desc = dataset_info.dataset_description.copy()
    for entry in desc:
        desc[entry]["type"] = int(desc[entry]["type"])
        desc[entry]["range"] = int(desc[entry]["range"])
        desc[entry]["lower_bound"] = int(desc[entry]["lower_bound"])
        desc[entry]["upper_bound"] = int(desc[entry]["upper_bound"])
        desc[entry]["upper_bound"] = int(desc[entry]["upper_bound"])
        desc[entry]["original_position"] = int(desc[entry]["original_position"])
        desc[entry]["current_position"] = int(desc[entry]["current_position"])
    out_str += json.dumps(desc, indent=4)
    inv_desc = dataset_info.inverse_dataset_description.copy()
    for entry in inv_desc:
        desc[entry]["type"] = int(desc[entry]["type"])
        desc[entry]["range"] = int(desc[entry]["range"])
        desc[entry]["lower_bound"] = int(desc[entry]["lower_bound"])
        desc[entry]["upper_bound"] = int(desc[entry]["upper_bound"])
        desc[entry]["upper_bound"] = int(desc[entry]["upper_bound"])
        desc[entry]["original_position"] = int(desc[entry]["original_position"])
        desc[entry]["current_position"] = int(desc[entry]["current_position"])
    out_str += json.dumps(desc, indent=4)
    out_str += "name: " + str(dataset_info.name) + "\n"
    out_str += "class_name: " + str(dataset_info.class_name) + "\n"
    return out_str


def numerical_list_tostr(arr) -> str:
    '''
    prints a list of float/double/int matching the cython formatting
    '''
    out_str = ""
    out_str += "["
    for i in range(len(arr)):
        out_str += str(arr[i])
        if i != len(arr)-1:
            out_str += ", "
    out_str += "]"
    return out_str


def Solution_tostr(solution: Solution):
    # cdef Solution solution = Solution(
    #     # cppvector[SplitPoint] conditions =
    #     cppvector[SplitPoint](),
    #     # cppvector[int] set_rules =
    #     cppvector[int](),
    #     # cppvector[double] instance =
    #     to_explain
    #     # double distance =
    #     -1,
    #     # bool found =
    #     False,
    #     # long num_evaluations =
    #     0,
    #     # long num_discovered_non_foil =
    #     0,
    #     # int estimated_class =
    #     -1
    # )
    out_str = ""
    out_str += "\tconditions:\n"
    for split in solution.conditions:
        out_str += "\t" + SplitPoint_tostr(split) + "\n"
    out_str += "\tset_rules: " + str(solution.set_rules) + "\n"
    out_str += "\tinstance: " + numerical_list_tostr(solution.instance) + "\n"
    out_str += "\tdistance: " + str(solution.distance) + "\n"
    out_str += "\tfound: " + str(solution.found) + "\n"
    out_str += "\tnum_evaluations: " + str(solution.num_evaluations) + "\n"
    out_str += "\tnum_discovered_non_foil: " + str(solution.num_discovered_non_foil) + "\n"
    out_str += "\testimated_class: " + str(solution.estimated_class) + "\n"
    return out_str
