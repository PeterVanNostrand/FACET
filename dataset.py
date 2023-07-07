import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List
from enum import Enum

from baselines.mace.fair_utils_data import get_one_hot_encoding as mace_get_one_hot
from baselines.mace.fair_adult_data import load_adult_data_new
from baselines.mace.loadData import loadDataset
from baselines.mace.loadData import DatasetAttribute

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
    "compas": "COMPAS-ProPublica_processedMACE",
    "credit": "Credit-Card-Default_processedMACE",
}

DS_DIMENSIONS = {
    "cancer": (568, 30),
    "glass": (162, 9),
    "magic": (19019, 10),
    "spambase": (4600, 57),
    "vertebral": (309, 6)
}


class FeatureType(Enum):
    Numeric = 1
    Binary = 2
    Discrete = 3
    Categorical = 4
    CategoricalNonOneHot = 5


class FeatureActionnability(Enum):
    Free = 1
    Fixed = 2
    Increasing = 3
    Predict = 4


def load_data(dataset_name, preprocessing: str = "Normalize"):
    '''
    Returns one of many possible anomaly detetion datasets based on the given `dataset_name`

    Parameters
    ----------
    dataset_name : the abbreviated name of the dataset to load, see README for all datset options
    preprocessing: if None do nothing, if Normalize normalize all features [0,1], if Scale Standardize
        features by removing the mean and scaling to unit variance

    Returns
    -------
    x : the dataset samples with shape (nsamples, nfeatures)
    y : the dataset labels with shape (nsamples,) containing 0 to represent a normal label and 1 for an anomaly label
    '''
    if False:
        pass
    elif dataset_name == "adult":
        x, y = util_load_adult()
    elif dataset_name == "cancer":
        x, y = util_load_cancer()
    elif dataset_name == "glass":
        x, y = util_load_glass()
    elif dataset_name == "magic":
        x, y = util_load_magic()
    elif dataset_name == "spambase":
        x, y = util_load_spambase()
    elif dataset_name == "vertebral":
        x, y = util_load_vertebral()
    else:
        print("ERROR NO SUCH DATASET")
        exit(0)

    if preprocessing == "Normalize":
        # normalize x to [0, 1]
        max_value = np.max(x, axis=0)
        min_value = np.min(x, axis=0)
        x = (x - min_value) / (max_value - min_value)
    elif preprocessing == "Scale":
        float_transformer = StandardScaler()
        x = float_transformer.fit_transform(x)
    return x, y


def type_to_enum(col_types) -> List[FeatureType]:
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


def action_to_enum(col_actionability) -> List[FeatureActionnability]:
    action_enums = []
    for action_str in col_actionability:
        if action_str == "FREE":
            action_enums.append(FeatureActionnability.Free)
        elif action_str == "FIXED":
            action_enums.append(FeatureActionnability.Fixed)
        elif action_str == "INC":
            action_enums.append(FeatureActionnability.Increasing)
        elif action_str == "PREDICT":
            action_enums.append(FeatureActionnability.Predict)
        elif action_str == "PROBLEM":
            print("problematic feature treated as free")
            action_enums.append(FeatureActionnability.Free)
        else:
            print("unknown actionnability: {}".format(action_str))
            return None
    return action_enums


# MACE VALUES
'''
VALID_ATTRIBUTE_DATA_TYPES = {
    'numeric-int',
    'numeric-real',
    'binary',
    'categorical',
    'sub-categorical',
    'ordinal',
    'sub-ordinal'}
VALID_ATTRIBUTE_NODE_TYPES = {
    'meta',
    'input',
    'output'}
VALID_ACTIONABILITY_TYPES = {
    'none',
    'any',
    'same-or-increase',
    'same-or-decrease'}
VALID_MUTABILITY_TYPES = {
    True,
    False}
'''

MACE_TYPE_FROM_FACET_TYPE = {
    FeatureType.Numeric: "numeric-real",
    FeatureType.Binary: "binary",
    FeatureType.Discrete: "numeric-int",
    FeatureType.Categorical: "categorical",
    FeatureType.CategoricalNonOneHot: "ordinal",  # ! WARNING NOT MACE SUPPORTED
}

MACE_ACTION_FROM_FACET_ACTION = {
    FeatureActionnability.Free: "any",
    FeatureActionnability.Fixed: "none",
    FeatureActionnability.Increasing: "same-or-increase",
    FeatureActionnability.Predict: "none",
}

MACE_MUTABILITY_FROM_FACET_ACTION = {
    FeatureActionnability.Free: True,
    FeatureActionnability.Fixed: False,
    FeatureActionnability.Increasing: True,
    FeatureActionnability.Predict: False,
}

'''
        attributes_non_hot[col_name] = DatasetAttribute(
            attr_name_long=col_name,
            attr_name_kurz='y',
            attr_type='binary',
            node_type='output',
            actionability='none',
            mutability=False,
            parent_name_long=-1,
            parent_name_kurz=-1,
            lower_bound=data_frame_non_hot[col_name].min(),
            upper_bound=data_frame_non_hot[col_name].max())
'''


def get_mace_non_hot(data, col_names, col_types, col_actions):
    '''
    Takes FACET/OCEAN encoded feature types and converts them to MACE DatasetAttributes
    '''
    attributes_non_hot = {}

    for i in range(len(col_names)):
        if col_types[i] in (FeatureType.Numeric, FeatureType.Binary, FeatureType.Discrete, FeatureType.CategoricalNonOneHot):
            col_name = col_names[i]

            attributes_non_hot[col_name] = DatasetAttribute(
                attr_name_long=col_name,
                attr_name_kurz=f"x{i}" if col_actions[i] != FeatureActionnability.Predict else "y",
                attr_type=MACE_TYPE_FROM_FACET_TYPE[col_types[i]],
                node_type="input" if col_actions[i] != FeatureActionnability.Predict else "output",
                actionability=MACE_ACTION_FROM_FACET_ACTION[col_actions[i]],
                mutability=MACE_MUTABILITY_FROM_FACET_ACTION[col_actions[i]],
                parent_name_long=-1,
                parent_name_kurz=-1,
                lower_bound=data[col_name].min(),
                upper_bound=data[col_name].max()
            )

    return attributes_non_hot


def util_load_adult():
    # load the csv
    path: str = DS_PATHS["adult"]
    data: pd.array = pd.read_csv(path)

    # get all columns names and types
    col_names: List[str] = list(data.columns)
    col_types = type_to_enum(list(data.iloc[0]))
    col_actionabiltiy = action_to_enum(list(data.iloc[1]))

    # separate the target column from the data
    target_col: int = col_actionabiltiy.index(FeatureActionnability.Predict)
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

    # get the unique possible values for categorial and discrete values
    for column in self.data:
        if column != 'Class':
            if self.featuresType[c] in [FeatureType.Discrete, FeatureType.Categorical]:
                self.featuresPossibleValues.append(self.data[column].unique())
            else:
                self.featuresPossibleValues.append([])

    X = []
    one_hot_mappings = {}
    one_hot_names = []

    # one-hot encode the input categorical features
    for i in range(len(input_col_types)):
        # non-categorical features don't need one-hot encoding, embed them directly
        if input_col_types[i] in [FeatureType.Numeric, FeatureType.Binary, FeatureType.Discrete, FeatureType.CategoricalNonOneHot]:
            X.append(input[:, i])
            one_hot_names.append(input_col_names[i].lower())
        # if its a categorical feature, one-hot encode it
        elif input_col_types[i] == FeatureType.Categorical:
            one_hot_i, dict_i = mace_get_one_hot(input[:, i])
            dict_i_inv = {val: key for key, val in dict_i.items()}
            # add a new column for each unique value
            for j in range(one_hot_i.shape[1]):
                X.append(one_hot_i[:, j])
                col_name = "{}_cat_{}".format(input_col_names[i], j)
                one_hot_mappings[col_name] = dict_i_inv[j]
                one_hot_names.append(col_name)

    mace_ds = loadDataset(dataset_name="adult", return_one_hot=False,
                          load_from_cache=False, debug_flag=True, my_df=data)
    mace_attrs = mace_ds.attributes_long

    # TODO: should I ignore feature actionability and ranges? will need to at least handle for categorical one-hots
    # TODO: 1. Determine how OCEAN handles min/max allowable values for discrete (integer) features and replicate this
    # TODO: 2. Get OCEAN running from our parsed data
    # TODO: 3. Get MACE running from our parsed data
    # TODO: 4. Implement feature typing and actionability to FACET
    # TODO: 5. Get RFOCSE working
    # TODO: 6. Get AFT working

    # prepare the MACE dataset info object
    non_hot_df = data.iloc[2:].astype(np.float64)
    attributes_non_hot = get_mace_non_hot(data=non_hot_df, col_names=col_names,
                                          col_types=col_types, col_actions=col_actionabiltiy)

    a = 2
    return None, None


def util_load_cancer():
    data = pd.read_csv(DS_PATHS["cancer"]).to_numpy()
    labels = data[:, 1:2].squeeze()  # lables of M=Malignant, B=Benign

    # label malignant as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "M"] = 1
    x = data[:, 2:].astype(float)
    return x, y


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
    return x, y


def util_load_magic():
    data = pd.read_csv(DS_PATHS["magic"]).to_numpy()
    labels = data[:, -1:].squeeze()  # lables of g=gamma (signal), h=hadron (background)

    # label gamma (signal) as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "g"] = 1
    x = data[:, :-1].astype(float)
    return x, y


def util_load_spambase():
    path = DS_PATHS["spambase"]
    data = pd.read_csv(path).to_numpy()
    x = data[:, :-1]
    y = data[:, -1:].squeeze().astype('int')
    return x, y


def util_load_vertebral():
    data = pd.read_csv(DS_PATHS["vertebral"], delim_whitespace=True).to_numpy()
    labels = data[:, -1:].squeeze()  # lables of g=gamma (signal), h=hadron (background)

    # label gamma (signal) as 1, benign as 0
    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "AB"] = 1
    x = data[:, :-1].astype(float)
    return x, y
