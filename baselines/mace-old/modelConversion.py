import numpy as np
from sklearn.tree import _tree, export_graphviz
from pysmt.shortcuts import *
from pysmt.typing import *


def tree2formula(tree, model_symbols, return_value='class_idx_max', tree_idx=''):
    tree_ = tree.tree_
    feature_names = list(model_symbols['counterfactual'].keys())
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = float(tree_.threshold[node])
            return Or(
                And(
                    LE(ToReal(model_symbols['counterfactual'][name]['symbol']), Real(threshold)),
                    recurse(tree_.children_left[node])
                ),
                And(
                    Not(LE(ToReal(model_symbols['counterfactual'][name]['symbol']), Real(threshold))),
                    recurse(tree_.children_right[node])
                )
            )
        else:
            if return_value == 'class_idx_max':
                values = list(tree_.value[node][0])
                output = bool(values.index(max(values)))
                return EqualsOrIff(model_symbols['output']['y']['symbol'], Bool(output))
            elif return_value == 'class_prob_array':
                prob_array = list(np.divide(tree_.value[node][0], np.sum(tree_.value[node][0])))
                return And(
                    EqualsOrIff(model_symbols['aux'][f'p0{tree_idx}']['symbol'], Real(float(prob_array[0]))),
                    EqualsOrIff(model_symbols['aux'][f'p1{tree_idx}']['symbol'], Real(float(prob_array[1])))
                )

    return recurse(0)


def forest2formula(forest, model_symbols):
    model_symbols['aux'] = {}
    for tree_idx in range(len(forest.estimators_)):
        model_symbols['aux'][f'p0{tree_idx}'] = {'symbol': Symbol(f'p0{tree_idx}', REAL)}
        model_symbols['aux'][f'p1{tree_idx}'] = {'symbol': Symbol(f'p1{tree_idx}', REAL)}

    tree_formulas = And([
        tree2formula(forest.estimators_[tree_idx], model_symbols, return_value='class_prob_array', tree_idx=tree_idx)
        for tree_idx in range(len(forest.estimators_))
    ])
    output_formula = Ite(
        GE(
            Plus([model_symbols['aux'][f'p0{tree_idx}']['symbol'] for tree_idx in range(len(forest.estimators_))]),
            Plus([model_symbols['aux'][f'p1{tree_idx}']['symbol'] for tree_idx in range(len(forest.estimators_))])
        ),
        EqualsOrIff(model_symbols['output']['y']['symbol'], FALSE()),
        EqualsOrIff(model_symbols['output']['y']['symbol'], TRUE()),
    )

    return And(
        tree_formulas,
        output_formula
    )
