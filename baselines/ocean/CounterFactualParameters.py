from enum import Enum
import gurobipy as gp
from gurobipy import GRB
import numpy as np

eps = 1e-5

# TreeConstraintsType = Enum("TreeConstraintsType", "BigM ExtendedFormulation")


class TreeConstraintsType(Enum):
    BigM = 1
    ExtendedFormulation = 2
    LinearCombinationOfPlanes = 3


class BinaryDecisionVariables(Enum):
    LeftRight_lambda = 1
    PathFlow_y = 2


class FeatureType(Enum):
    Numeric = 1
    Binary = 2
    Discrete = 3
    Categorical = 4
    CategoricalNonOneHot = 5


class FeatureActionability(Enum):
    Free = 1
    Fixed = 2
    Increasing = 3
    Predict = 4


def getFeatureType(name):
    if name == 'N':
        return FeatureType.Numeric
    elif name == 'B':
        return FeatureType.Binary
    elif name == 'D':
        return FeatureType.Discrete
    elif name == 'C':
        return FeatureType.Categorical
    else:
        print("Unknown feature type", name)
        return None


def getFeatureType_oldDatasetsWithoutActionnability(name):
    if name == 'C':
        return FeatureType.Numeric
    elif name == 'B':
        return FeatureType.Binary
    elif name == 'D':
        return FeatureType.Discrete
    else:
        print("Unknown feature type", name)
        return None


def isFeatureTypeScalable(featureType):
    if featureType == FeatureType.Categorical:
        return False
    return True


def getFeatureActionnability(name):
    if name == "FREE":
        return FeatureActionability.Free
    elif name == "FIXED":
        return FeatureActionability.Fixed
    elif name == "INC":
        return FeatureActionability.Increasing
    elif name == "PREDICT":
        return FeatureActionability.Predict
    elif name == "PROBLEM":
        print("Problematic feature treated as free")
        return FeatureActionability.Free
    else:
        print("Unknown actionnability", name)
        return None


class ObjectiveType(Enum):
    L0 = 0
    L1 = 1
    L2 = 2
    # Mahalanobis1 = 11
