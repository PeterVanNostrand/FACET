from collections import defaultdict, namedtuple
from operator import itemgetter

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .datasets import DatasetInfo

Node = namedtuple('Node', ('feature', 'threshold', 'left', 'right'))
Leaf = namedtuple('Leaf', ('values', 'node_id'))
Rule = namedtuple('Rule', ('conditions', 'label', 'node_id', 'area'))
Condition = namedtuple('Condition', ('feature', 'threshold', 'is_leq'))


def convert_dt(dt: DecisionTreeClassifier):
    n_nodes = dt.tree_.node_count
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    values = dt.tree_.value

    nodes = {}
    for node_id in reversed(range(n_nodes)):
        if children_left[node_id] != children_right[node_id]:
            nodes[node_id] = Node(feature[node_id], threshold[node_id], nodes[children_left[node_id]],
                                  nodes[children_right[node_id]])
        else:
            nodes[node_id] = Leaf(values[node_id], node_id)

    return nodes[0]


def extract_rules(dt, dataset_info):
    def _convert(node, prev_conds):
        if isinstance(node, Leaf):
            return [Rule(conditions=prev_conds, label=node.values, node_id=node.node_id,
                         area=calculate_rule_size(prev_conds, dataset_info))]
        else:
            rules_left = _convert(node.left, prev_conds + [Condition(node.feature, node.threshold, is_leq=True)])
            rules_right = _convert(node.right, prev_conds + [Condition(node.feature, node.threshold, is_leq=False)])
            return rules_left + rules_right

    return _convert(dt, [])


def calculate_rule_size(conditions, dataset_info):
    n_features = len(dataset_info.dataset_description)
    area = n_features

    for (feat, is_leq), threshold in extract_rule_condition_set(conditions).items():
        feat_into = dataset_info.dataset_description[feat]
        lb = feat_into['lower_bound']
        ub = feat_into['upper_bound']
        frange = feat_into['range']

        if feat_into['type'] != 4:
            if is_leq:
                condition_area = (ub - threshold) / frange
            else:
                condition_area = (threshold - lb) / frange

            area -= condition_area

    return area


def extract_rule_condition_set(conditions):
    condition_set = {}
    for condition in conditions:
        condition_id = (condition.feature, condition.is_leq)
        if condition_id in condition_set:
            other_threshold = condition_set[condition_id]
            if condition.is_leq and other_threshold > condition.threshold:
                condition_set[condition_id] = condition.threshold
            elif not condition.is_leq and other_threshold < condition.threshold:
                condition_set[condition_id] = condition.threshold
        else:
            condition_set[condition_id] = condition.threshold

    return condition_set


def convert_rf_format(sklearn_rf, dataset_info: DatasetInfo):
    rules = []
    feats_conditions = {}
    rule_len = {}
    labels = {}
    rule_tree_mapping = {}

    for t_id, dt in enumerate(sklearn_rf.estimators_):
        tree_rules = extract_rules(convert_dt(dt), dataset_info)
        rules.extend(zip([t_id] * len(tree_rules), tree_rules))

    for attr, attr_info in dataset_info.dataset_description.items():
        if attr_info['type'] in (1, 2, 3):
            feat_conditions = {
                'type': attr_info['type'],
                'side_1': defaultdict(list),
                'side_2': defaultdict(list),
            }
        else:
            feat_conditions = {
                'type': 4,
                'side_1': {i: [] for i in range(len(attr_info['categories_original_position']))},
                'side_2': {i: [] for i in range(len(attr_info['categories_original_position']))}
            }

        feats_conditions[attr] = feat_conditions

    sklearn_rule_mapping = defaultdict(dict)
    for rule_id, (t_id, rule) in enumerate(rules):
        labels[rule_id] = rule.label
        sklearn_rule_mapping[t_id][rule.node_id] = rule_id

        rule_tree_mapping[rule_id] = t_id
        condition_set = extract_rule_condition_set(rule.conditions)

        for (feat_name, is_leq), threshold in condition_set.items():
            feat_new_idx = dataset_info.inverse_dataset_description[feat_name]['current_position']
            attr_info = dataset_info.dataset_description[feat_new_idx]

            if attr_info['type'] in (1, 2, 3):
                insert_in = 'side_1' if is_leq else 'side_2'
                feats_conditions[feat_new_idx][insert_in][threshold].append(rule_id)
            else:
                insert_in = 'side_1' if is_leq else 'side_2'
                bucket = dataset_info.inverse_dataset_description[feat_name]['category_index']
                feats_conditions[feat_new_idx][insert_in][bucket].append(rule_id)

        rule_len[rule_id] = len(condition_set)

    rule_probs = np.array([label.reshape(-1) / np.sum(label, axis=-1) for rule_id, label in labels.items()])

    return ParsedRF(sklearn_rf, len(sklearn_rf.estimators_), sklearn_rf.n_classes_, rule_probs,
                    rule_tree_mapping, feats_conditions, rules, sklearn_rule_mapping)


ParsedRF = namedtuple('ParsedRF',
                      ['sklearn_model', 'n_trees', 'n_labels', 'rule_probs', 'rule_tree', 'feats_conditions', 'rules',
                       'sklearn_rule_mapping'])


def export(rules, to_file, dataset_info):
    tree_rules = defaultdict(list)

    for i, (t_id, rule) in enumerate(rules):
        tree_rules[t_id].append((i, rule))

    with open(to_file, 'w') as fout:
        for tid, tree_rules in tree_rules.items():
            fout.write('Tree {} \n'.format(tid))
            for rule_id, tree_rule in tree_rules:
                pretty_rule = prettify_rule(tree_rule, dataset_info)
                fout.write('\t Rule {} [{}] |'.format(rule_id, str(tree_rule.label)) + str(pretty_rule) + '\n')


def prettify_rule(rule, dataset_info):
    rule_set = {
        attr: {
            'type': attr_info['type'],
            'values': [None, None] if attr_info['type'] in (1, 2) else list(
                attr_info.get('categories_original_position',
                              [0, 1])),
            'modified': False
        }
        for attr, attr_info in dataset_info.dataset_description.items()
    }

    for feature, threshold, is_leq in rule.conditions:
        inv_attr = dataset_info.inverse_dataset_description[feature]['current_position']
        attr_info = dataset_info.dataset_description[inv_attr]
        feature_ruleset = rule_set[inv_attr]
        feature_ruleset['modified'] = True
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

    return {k: v['values'] for k, v in rule_set.items() if v['modified']}
