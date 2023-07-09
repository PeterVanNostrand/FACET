from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple

from baselines.mace.fair_utils_data import get_one_hot_encoding as mace_get_one_hot
from baselines.ocean.CounterFactualParameters import FeatureActionability
from baselines.ocean.CounterFactualParameters import FeatureType
from baselines.ocean.CounterFactualParameters import BinaryDecisionVariables, TreeConstraintsType
from baselines.ocean.RandomForestCounterFactual import RandomForestCounterFactualMilp
from manager import MethodManager
from utilities.metrics import classification_metrics

# a list of the abbreviated name for all classification datasets
DS_NAMES = [
    "cancer",
    "glass",
    "magic",
    "spambase",
    "vertebral",
    "adult",
    "compas",
]

DS_PATHS = {
    "cancer": "data/cancer/wdbc.data",
    "glass": "data/glass/glass.data",
    "magic": "data/magic/magic04.data",
    "spambase": "data/spambase/spambase.data",
    "vertebral": "data/vertebral/column_2C.dat",
    "adult": "data/adult/Adult_processedMACE.csv",
    "compas": "data/compas/COMPAS-ProPublica_processedMACE.csv",
    "credit": "data/credit/Credit-Card-Default_processedMACE.csv",
}

DS_DIMENSIONS = {
    "cancer": (568, 30),
    "glass": (162, 9),
    "magic": (19019, 10),
    "spambase": (4600, 57),
    "vertebral": (309, 6)
}


@dataclass
class DataInfo:
    col_names: List[str]
    col_types: List[FeatureType]
    col_actions: List[FeatureActionability]
    possible_vals: List
    one_hot_schema: dict

    def __post_init__(self):
        self.ncols = len(self.col_names)

    @classmethod
    def generic(cls, ncols):
        '''
        Creates a generic DataInfo object with all numeric, fully actionable and unbounded features. Not for use with non-numeric or one-hot encoded features
        '''
        col_names = ["x{}".format(_) for _ in range(ncols)]
        col_types = [FeatureType.Numeric for _ in range(ncols)]
        col_actionability = [FeatureActionability.Free for _ in range(ncols)]
        possible_values = [[] for _ in range(ncols)]
        one_hot_schema = {}

        return cls(col_names, col_types, col_actionability, possible_values, one_hot_schema)


def load_data(dataset_name, preprocessing: str = "Normalize"):
    '''
    Returns one of many possible anomaly detetion datasets based on the given `dataset_name`. Note that all features are currently treated as actionable regardless of source file designation

    Parameters
    ----------
    dataset_name : the abbreviated name of the dataset to load, see README for all datset options
    preprocessing: if None do nothing, if Normalize normalize all features [0,1], if Scale Standardize
        features by removing the mean and scaling to unit variance

    Returns
    -------
    x : the dataset samples with shape (nsamples, nfeatures)
    y : the dataset labels with shape (nsamples,) containing 0 to represent a normal label and 1 for an anomaly label
    ds_info: a DataInfo object containing information about the feature types, one-hot encoding, actionability, etc
    '''
    if False:
        pass
    elif dataset_name in ["adult", "compas", "credit"]:
        x, y, ds_info = load_facet_data(dataset_name)
    elif dataset_name == "cancer":
        x, y, ds_info = util_load_cancer()
    elif dataset_name == "glass":
        x, y, ds_info = util_load_glass()
    elif dataset_name == "magic":
        x, y, ds_info = util_load_magic()
    elif dataset_name == "spambase":
        x, y, ds_info = util_load_spambase()
    elif dataset_name == "vertebral":
        x, y, ds_info = util_load_vertebral()
    else:
        print("ERROR NO SUCH DATASET")
        exit(0)

    # normalize numeric and discrete features to [0, 1]
    if preprocessing == "Normalize":
        normalize_data(x, ds_info.col_types)
    return x, y, ds_info


def type_to_enum(col_types) -> List[FeatureType]:
    '''
    Convert the characters `N`, `B`, `D`, `C` used to denote feature types in the dataset csv to their matching FeatureType enum value.
    '''
    enum_types = []
    for type_str in col_types:
        if type_str == "N":
            enum_types.append(FeatureType.Numeric)
        elif type_str == "B":
            enum_types.append(FeatureType.Binary)
        elif type_str == "D":
            enum_types.append(FeatureType.Discrete)
        elif type_str == "C":
            enum_types.append(FeatureType.Categorical)
        else:
            print("unkown feature type: {}".format(type_str))
            return None
    return enum_types


def action_to_enum(col_actionability) -> List[FeatureActionability]:
    '''
    Converts string values for action types to the corresponding FeatureActionability enum value
    '''
    action_enums = []
    for action_str in col_actionability:
        if action_str == "FREE":
            action_enums.append(FeatureActionability.Free)
        elif action_str == "FIXED":
            action_enums.append(FeatureActionability.Fixed)
        elif action_str == "INC":
            action_enums.append(FeatureActionability.Increasing)
        elif action_str == "PREDICT":
            action_enums.append(FeatureActionability.Predict)
        elif action_str == "PROBLEM":
            print("problematic feature treated as free")
            action_enums.append(FeatureActionability.Free)
        else:
            print("unknown actionnability: {}".format(action_str))
            return None
    return action_enums


def normalize_data(x: np.ndarray, col_types: List[FeatureType]):
    '''
    Normalizes Numeric and Discrete columns to the range [0, 1]
    '''
    for i in range(len(col_types)):
        if col_types[i] in [FeatureType.Numeric, FeatureType.Discrete]:
            max_value = np.max(x[:, i])
            min_value = np.min(x[:, i])
            x[:, i] = (x[:, i] - min_value) / (max_value - min_value)
    return x


def one_hot_encode(input: np.ndarray, col_names: List[str], col_types: List[FeatureType]) -> DataInfo:
    '''
    One-hot encode the given data. Categorical features are removed and replaced with one column for each categorical value

    `input`: a numpy array of data containing categorical features
    `col_names`: a list of the names of each feature in `input`
    `col_types`: a list of the type of each feature in `input` only FeatureType.Categorical will be one-hot encoded
    '''
    x = []
    one_hot_mappings = {}
    one_hot_names = []
    one_hot_types = []
    one_hot_schema = {}  # dict to trace one-hot from source col to column index

    # one-hot encode the input categorical features
    for i in range(len(col_types)):
        # non-categorical features don't need one-hot encoding, embed them directly
        if col_types[i] in [FeatureType.Numeric, FeatureType.Binary, FeatureType.Discrete, FeatureType.CategoricalNonOneHot]:
            x.append(input[:, i])
            one_hot_names.append(col_names[i])
            one_hot_types.append(col_types[i])
        # if its a categorical feature, one-hot encode it
        elif col_types[i] == FeatureType.Categorical:
            one_hot_schema[col_names[i]] = []
            one_hot_i, dict_i = mace_get_one_hot(input[:, i])
            dict_i_inv = {val: key for key, val in dict_i.items()}
            # add a new column for each unique value
            for j in range(one_hot_i.shape[1]):
                x.append(one_hot_i[:, j])
                col_name = "{}_{}".format(col_names[i], j)
                one_hot_mappings[col_name] = dict_i_inv[j]
                one_hot_names.append(col_name)
                one_hot_types.append(FeatureType.Binary)
                one_hot_schema[col_names[i]].append(len(x) - 1)
    x = np.array(x).T

    # get the allowed values for each feature
    one_hot_possible_vals = get_possible_vals(x, one_hot_types)

    # allowing alteration on all features
    one_hot_actions = [FeatureActionability.Free for _ in range(len(one_hot_names))]

    # create the data info object
    ds_info = DataInfo(one_hot_names, one_hot_types, one_hot_actions, one_hot_possible_vals, one_hot_schema)
    return x, ds_info


def get_possible_vals(x, col_types):
    '''
    Determine the range of allowed values for each column. Currently only applying a range to Discrete values as  this is reqruired by OCEAN
    '''
    possible_vals = [[] for _ in range(len(col_types))]
    for i in range(len(col_types)):
        if col_types[i] == FeatureType.Discrete:
            possible_vals[i] = [np.min(x[:, i]), np.max(x[:, i])]
    return possible_vals


def load_facet_data(ds_name: str) -> Tuple[np.ndarray, np.ndarray, DataInfo]:
    '''
    Load a csv file tagged with feature types and actionabilities as per the FACET/OCEAN encoding

    Row 1: column names
    Row 2: feature types using the character encodings `N`, `B`, `D`, `C`. See type_to_enum()
    Row 3: feature actionability `FREE`, `FIXED`, `INC`, `PREDICT` or `PROBLEM`. See action_to_enum()
    '''
    # load the csv
    path: str = DS_PATHS[ds_name]
    data: pd.array = pd.read_csv(path)

    # get all columns names and types
    col_names: List[str] = list(data.columns)
    col_types = type_to_enum(list(data.iloc[0]))
    col_actionabiltiy = action_to_enum(list(data.iloc[1]))

    # separate the target column from the data
    target_col: int = col_actionabiltiy.index(FeatureActionability.Predict)
    target_col_name: str = col_names[target_col]
    target = data[target_col_name].iloc[2:].to_numpy(dtype=np.float64)

    # separate the input columns from the data
    input_col_names = col_names.copy()
    input_col_names.pop(target_col)
    input_col_types = col_types.copy()
    input_col_types.pop(target_col)
    input_col_action = col_actionabiltiy.copy()
    input_col_action.pop(target_col)
    input = data[input_col_names].iloc[2:].to_numpy(dtype=np.float64)
    input = normalize_data(input, input_col_types)

    x, ds_info = one_hot_encode(input, input_col_names, input_col_types)
    y = target.astype(int)

    return x, y, ds_info


def test_ocean_one_hot(x: np.ndarray, y: np.ndarray, ds_info: DataInfo) -> None:
    random_state = 0
    random.seed(random_state)
    np.random.seed(random_state)

    # split the training and testing data
    n_explain = 10
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=random_state)
    if n_explain is not None:
        x_explain = xtest[:n_explain]
        y_explain = ytest[:n_explain]
    else:
        x_explain = xtest
        y_explain = ytest
        n_explain = x_explain.shape[0]

    # train the ensemble
    explainer = "OCEAN"
    from experiments.experiments import DEFAULT_PARAMS
    params = DEFAULT_PARAMS
    manager = MethodManager(explainer=explainer, hyperparameters=params, random_state=random_state)
    manager.train(xtrain, ytrain)

    # get the samples to explain
    preds = manager.predict(x_explain)
    accuracy, precision, recall, f1 = classification_metrics(preds, y_explain, verbose=False)
    counterfactual_classes = ((preds - 1) * -1)

    # prepare an array to hold the explanations
    xprime = np.empty(shape=x_explain.shape)
    xprime[:, :] = np.inf

    # explain samples one at a time
    for i in range(x_explain.shape[0]):
        # shape data for OCEAN to oad
        to_explain = x_explain[i].copy()
        sample = [pd.Series(to_explain, dtype=np.float64, name=str(i))]
        desired_label = counterfactual_classes[i]
        sample[0].index = ds_info.col_names

        # give ocean info on the feature type, values, and actionability
        # !WARNING: Ignoring feature actionability constraints from data files
        feat_types = ds_info.col_types
        feat_actionability = ds_info.col_types

        # explain the sample
        randomForestMilp = RandomForestCounterFactualMilp(
            classifier=manager.random_forest.model,
            sample=sample,
            outputDesired=desired_label,
            isolationForest=None,
            constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
            objectiveNorm=2,
            mutuallyExclusivePlanesCutsActivated=True,
            strictCounterFactual=True,
            verbose=False,
            featuresType=feat_types,
            featuresPossibleValues=ds_info.possible_vals,
            featuresActionnability=feat_actionability,
            oneHotEncoding=ds_info.one_hot_schema,
            binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
            randomCostsActivated=False
        )
        randomForestMilp.buildModel()
        randomForestMilp.solveModel()
        xprime[i] = np.array(randomForestMilp.x_sol[0])

    # check that all counterfactuals result in a different class
    preds = manager.predict(xprime)
    failed_explanation = (preds != counterfactual_classes)
    xprime[failed_explanation] = np.tile(np.inf, x_explain.shape[1])


def check_one_hot_validity(x: np.ndarray, one_hot_schema: dict, verbose: bool = False) -> bool:
    '''
    Given an array `x` with one hot encoded features. Check that each one hot feature has exactly one column set

    `x`: a numpy array of shape `(n_instances, n_features)`
    `one_hot_schema`: a dictionary which maps `{"feature_name": [i, j, ... , k]}` where `feature_name` is the name of the feature which was one hot encoded and `[i, j, ... , k]` are the column indices for the resulting one hot columns
    '''
    all_valid = True
    for i in range(x.shape[0]):
        for cat_column_name, sub_col_idxs in one_hot_schema.items():
            n_set_colums = sum(x[i][sub_col_idxs])
            if (n_set_colums != 1):
                all_valid = False
                if verbose:
                    print("failed one-hot encoding\t\t{}\t{}\t{}".format(i,
                          cat_column_name, sum(x[i][sub_col_idxs])))
    return all_valid


def util_load_cancer():
    data = pd.read_csv(DS_PATHS["cancer"]).to_numpy()
    labels = data[:, 1:2].squeeze()  # lables of M=Malignant, B=Benign

    # label malignant as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "M"] = 1
    x = data[:, 2:].astype(float)

    ds_info = DataInfo.generic(ncols=x.shape[1])
    return x, y, ds_info


def util_load_glass():
    data = pd.read_csv(DS_PATHS["glass"]).to_numpy()
    labels = data[:, -1:].squeeze()

    # glass has 6 classes which correspond to six types of glass
    # 1 building_windows_float_processed
    # 2 building_windows_non_float_processed
    # 3 vehicle_windows_float_processed
    # 4 vehicle_windows_non_float_processed (none in this database)
    # 5 containers
    # 6 tableware
    # 7 headlamps
    # We aim to train a binary classifier to differentiate between float and nonfloat glass
    float_classes = [1, 3]
    nonfloat_classes = [2, 4]

    is_float = np.isin(labels, float_classes)
    is_nonfloat = np.isin(labels, nonfloat_classes)

    # label float as 1, nonfloat as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[is_float] = 1

    x = data[:, 1:-1].astype(float)

    # drop all samples that are not float or nonfloat (keeps ~75% of dataset)
    y = y[is_float | is_nonfloat]
    x = x[is_float | is_nonfloat]

    ds_info = DataInfo.generic(ncols=x.shape[1])
    return x, y, ds_info


def util_load_magic():
    data = pd.read_csv(DS_PATHS["magic"]).to_numpy()
    labels = data[:, -1:].squeeze()  # lables of g=gamma (signal), h=hadron (background)

    # label gamma (signal) as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "g"] = 1
    x = data[:, :-1].astype(float)
    ds_info = DataInfo.generic(ncols=x.shape[1])
    return x, y, ds_info


def util_load_spambase():
    path = DS_PATHS["spambase"]
    data = pd.read_csv(path).to_numpy()
    x = data[:, :-1]
    y = data[:, -1:].squeeze().astype('int')
    ds_info = DataInfo.generic(ncols=x.shape[1])
    return x, y, ds_info


def util_load_vertebral():
    data = pd.read_csv(DS_PATHS["vertebral"], delim_whitespace=True).to_numpy()
    labels = data[:, -1:].squeeze()  # lables of g=gamma (signal), h=hadron (background)

    # label gamma (signal) as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "AB"] = 1
    x = data[:, :-1].astype(float)
    ds_info = DataInfo.generic(ncols=x.shape[1])
    return x, y, ds_info
