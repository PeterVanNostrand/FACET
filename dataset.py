import copy
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.mace.fair_utils_data import get_one_hot_encoding as mace_get_one_hot
from baselines.ocean.CounterFactualParameters import FeatureActionability, FeatureType
from explainers.bit_vector import LOWER, UPPER

script_directory = os.path.dirname(os.path.abspath(__file__))

# a list of the abbreviated name for all classification datasets
DS_NAMES = [
    "cancer",
    "glass",
    "magic",
    "spambase",
    "vertebral",
    "adult",
    "compas",
    "loans",
]


DS_PATHS = {
    "cancer": os.path.join(script_directory, "data/cancer/wdbc.data"),
    "glass": os.path.join(script_directory, "data/glass/glass.data"),
    "magic": os.path.join(script_directory, "data/magic/magic04.data"),
    "spambase": os.path.join(script_directory, "data/spambase/spambase.data"),
    "vertebral": os.path.join(script_directory, "data/vertebral/column_2C.dat"),
    "adult": os.path.join(script_directory, "data/adult/Adult_processedMACE.csv"),
    "compas": os.path.join(script_directory, "data/compas/COMPAS-ProPublica_processedMACE.csv"),
    "credit": os.path.join(script_directory, "data/credit/Credit-Card-Default_processedMACE.csv"),
    "loans": os.path.join(script_directory, "data/loans/loans_continuous.csv"),
}

DS_DIMENSIONS = {
    "cancer": (568, 30),
    "glass": (162, 9),
    "magic": (19019, 10),
    "spambase": (4600, 57),
    "vertebral": (309, 6),
    "loans": (578, 4),
}

RANGED_DISCRETE = {
    # adult
    "Age": [0, 120],
    "HoursPerWeek": [0, 100],
    # credit
    "MonthsWithZeroBalanceOverLast6Months": [0, 6],
    "MonthsWithLowSpendingOverLast6Months": [0, 6],
    "MonthsWithHighSpendingOverLast6Months": [0, 6],
    # "MaxBillAmountOverLast6Months": [0, 50810],
    # "MaxPaymentAmountOverLast6Months": [0, 51430],
    # compas
    "PriorsCount": [0, 40]
}


@dataclass
class DataInfo:
    # init required
    col_names: list[str]
    col_types: list[FeatureType]
    col_actions: list[FeatureActionability]
    one_hot_schema: dict
    # computed in post_init()
    ncols: int = -1
    all_numeric: bool = False
    reverse_one_hot_schema = None
    # set in get_possible_vals
    possible_vals: list = None
    col_scales: dict = None
    # set by normalize()
    normalize_numeric: bool = False
    normalize_discrete: bool = False
    numeric_int_map: dict = None  # only used for adult on MACE
    col_to_idx: dict = None

    def __post_init__(self):
        self.ncols = len(self.col_names)
        # check if all the features are numeric
        self.all_numeric = True
        for type in self.col_types:
            if type != FeatureType.Numeric:
                self.all_numeric = False
        # create a reverse lookup for the schema
        self.reverse_one_hot_schema = {}
        for col_name, col_idxs in self.one_hot_schema.items():
            for idx in col_idxs:
                self.reverse_one_hot_schema[idx] = col_name

    def _map_cols(self):
        col_to_idx = dict()
        for i in range(self.ncols):
            col_to_idx["x{}".format(i)] = i
        self.col_to_idx = col_to_idx

    @classmethod
    def generic(cls, ncols):
        '''
        Creates a generic DataInfo object with all numeric, fully actionable and unbounded features. Not for use with non-numeric or one-hot encoded features
        '''
        col_names = ["x{}".format(_) for _ in range(ncols)]
        col_types = [FeatureType.Numeric for _ in range(ncols)]
        col_actionability = [FeatureActionability.Free for _ in range(ncols)]
        one_hot_schema = {}
        return cls(col_names, col_types, col_actionability, one_hot_schema)

    def copy(self):
        names = copy.deepcopy(self.col_names)
        types = copy.deepcopy(self.col_types)
        actions = copy.deepcopy(self.col_actions)
        schema = copy.deepcopy(self.one_hot_schema)
        new_info = DataInfo(names, types, actions, schema)

        new_info.possible_vals = copy.deepcopy(self.possible_vals)
        new_info.col_scales = copy.deepcopy(self.col_scales)
        new_info.normalize_numeric = copy.deepcopy(self.normalize_numeric)
        new_info.normalize_discrete = copy.deepcopy(self.normalize_discrete)
        new_info.numeric_int_map = copy.deepcopy(self.numeric_int_map)
        return new_info

    def get_possible_vals(self, x: np.ndarray, do_convert=False) -> None:
        '''
        Determine the range of allowed values for each column. For Discrete values this is a list of all allowed options, for Numeric values it is the min and max allowed value from the data. For Binary it is [0, 1]. Categorical features are one-hot encoded as multiple Binary columns
        '''
        possible_vals = [[] for _ in range(self.ncols)]  # a list of all possible values, may be scaled later
        col_scales = dict()  # a dict of the min and max allowed value, will never be scaled
        for i in range(self.ncols):
            col_name = self.col_names[i]
            col_type = self.col_types[i]
            col_min, col_max = [x[:, i].min(), x[:, i].max()]
            # binary features are always 0 or 1
            if col_type == FeatureType.Binary:
                possible_vals[i] = [0.0, 1.0]
            # numeric features can vary between a min and max
            if col_type == FeatureType.Numeric:
                # mace fails to handle numeric-real features with other featuers, this occurs for two columns in the adult dataset. in this case we map the values to ints and back
                if col_name in ["CapitalGain", "CapitalLoss"] and do_convert:
                    # recast the feature as an int feature, will handle later
                    self.col_types[i] = FeatureType.Discrete
                    if self.numeric_int_map is None:
                        self.numeric_int_map = {}
                    unique_vals = list(np.unique(x[:, i]))
                    self.numeric_int_map[col_name] = unique_vals
                    col_min = 0
                    col_max = len(unique_vals)-1
                    for j in range(x.shape[0]):
                        x[j][i] = unique_vals.index(x[j][i])
                # set the possible vals
                possible_vals[i] = [col_min, col_max]
            # discrete features must be one of the allowed options (integers) between the min and max
            if col_type == FeatureType.Discrete:
                # if we need to override with a range
                if self.col_names[i] in RANGED_DISCRETE:
                    col_min, col_max = RANGED_DISCRETE[self.col_names[i]]
                    possible_vals[i] = list(range(col_min, col_max+1))
                # otherwise use the availible options
                else:
                    possible_vals[i] = list(np.unique(x[:, i]))
            # save the unscaled min/max along each column
            col_scales[i] = [col_min, col_max]
        self.possible_vals = possible_vals
        self.col_scales = col_scales

    def scale_points(self, x: np.ndarray):
        """
        Unscales the given points, x must be an array/list of shape (ninstances, ncols)
        """
        # create a 2D array copy of x
        scaled = np.array([x.copy()]) if len(x.shape) == 1 else x.copy()
        # normalize the copy
        for i in range(self.ncols):
            col_type = self.col_types[i]
            min_value, max_value = self.col_scales[i]
            # if we need to scale this column
            if (col_type == FeatureType.Numeric and self.normalize_numeric) or \
                    (col_type == FeatureType.Discrete and self.normalize_discrete):
                # scale the given data
                scaled[:, i] = (scaled[:, i] - min_value) / (max_value - min_value)
        # return the scaled copy
        return scaled

    def rescale_rects(self, rects: np.ndarray, scale_down: bool) -> np.ndarray:
        """
        Scales/unscales the given rectangles

        Args:
            rects (np.ndarray): an array/list of shape (nrects, ncols, 2)
            is_scale (bool): Whether to scale down (true) or up (false)
        """
        # create a copy of the rects
        rescaled_rects: np.ndarray = rects.copy()
        # if we only have one rect, add a dimension to allow for slicing
        if len(rescaled_rects.shape) == 2:
            rescaled_rects = rescaled_rects.reshape(-1, self.ncols, 2)
        # rescale all the rects
        for i in range(len(rescaled_rects)):  # for each rectangle
            for col_id in range(self.ncols):  # for each column
                col_type = self.col_types[col_id]  # get the column type
                # if we need to rescale this column
                if (col_type == FeatureType.Numeric and self.normalize_numeric) or \
                        (col_type == FeatureType.Discrete and self.normalize_discrete):
                    # get the min and max values
                    min_val, max_val = self.col_scales[col_id]
                    for end in [LOWER, UPPER]:  # rescale both rect ends
                        if scale_down:  # if we are scaling down e.g. 10-->1
                            rescaled_rects[i][col_id][end] = (
                                rescaled_rects[i][col_id][end] - min_val
                            ) / (max_val - min_val)
                        else:  # if we are scaling up, e.g., 1->10
                            rescaled_rects[i][col_id][end] = (
                                rescaled_rects[i][col_id][end] * (max_val - min_val) + min_val
                            )
        return rescaled_rects

    def normalize_data(self, x: np.ndarray, normalize_numeric: bool = True, normalize_discrete: bool = False) -> None:
        '''
        Normalizes Numeric and Discrete columns to the range [0, 1]. Updates `self.possible_vals` to match new scale
        '''
        # TODO: separate the process of picking col ranges from the process of normalizing
        self.normalize_numeric = normalize_numeric
        self.normalize_discrete = normalize_discrete
        for i in range(self.ncols):
            col_type = self.col_types[i]
            min_value, max_value = self.col_scales[i]
            # if we need to scale this column
            if (col_type == FeatureType.Numeric and normalize_numeric) or \
                    (col_type == FeatureType.Discrete and normalize_discrete):
                # scale the given data
                x[:, i] = (x[:, i] - min_value) / (max_value - min_value)
                # adjust the posssible values to match
                for j in range(len(self.possible_vals[i])):
                    self.possible_vals[i][j] = (self.possible_vals[i][j] - min_value) / (max_value - min_value)

    def unscale(self, x: np.ndarray, col_id: int = None):
        '''
        Unscales the given data to match the values ranges before normalization

        Parameters
        ----------
        x: a numpy array of shape (ncols,) or a single column value e.g. array of (1,) or a int/float
        col_id: when x is a single value, the integer column ID of the value to unscale. set to None otherwise

        Returns
        -------
        unscaled: the unscaled data matching the type and shape of x
        '''
        # if not self.normalize_numeric:
        #     return x

        # given a single value, unscale that
        if col_id is not None:
            col_type = self.col_types[col_id]
            if (col_type == FeatureType.Discrete and self.normalize_discrete) or \
                    (col_type == FeatureType.Numeric and self.normalize_numeric):
                return x * (self.col_scales[col_id][1] - self.col_scales[col_id][0]) + self.col_scales[col_id][0]
            else:
                return x
        # given an array of values, unscale them
        unscaled = x.copy()
        if len(unscaled.shape) == 1:  # given a single instance array (ncols,)
            for col_id in range(self.ncols):
                if col_id in self.col_scales:
                    col_type = self.col_types[col_id]
                    if (col_type == FeatureType.Discrete and self.normalize_discrete) or \
                            (col_type == FeatureType.Numeric and self.normalize_numeric):
                        min_val, max_val = self.col_scales[col_id]
                        unscaled[col_id] = unscaled[col_id] * (max_val - min_val) + min_val
        elif len(unscaled.shape) == 2:  # given an array of (ninstances, ncols)
            for i in range(unscaled.shape[0]):
                for col_id in range(self.ncols):
                    if col_id in self.col_scales:
                        col_type = self.col_types[col_id]
                        if (col_type == FeatureType.Discrete and self.normalize_discrete) or \
                                (col_type == FeatureType.Numeric and self.normalize_numeric):
                            min_val, max_val = self.col_scales[col_id]
                            unscaled[i, col_id] = unscaled[i, col_id] * (max_val - min_val) + min_val
        else:
            unscaled = None
            print("mishapen array passed to unscale()")
        return unscaled

    def check_valid(self, x: np.ndarray) -> bool:
        '''
        Checks if the given data array `x` matches requirements for one-hot encoding, discrete values, and value ranges

        Parameters
        ----------
        x: a scaled numpy array of shape (ninstances, ncols)

        Returns
        -------
        all_valid: a bool which is true iff all instances are valid
        '''
        all_valid = True
        for instance in x:
            valid_i = True
            for col_id in range(self.ncols):
                if self.col_types[col_id] == FeatureType.Binary:
                    # binary values must be either zero or one
                    valid_i = valid_i and (instance[col_id] in [0.0, 1.0])
                    # if the binary value encodes a one-hot feature
                    if col_id in self.reverse_one_hot_schema:
                        feat_name = self.reverse_one_hot_schema[col_id]
                        # there can only be column set hot
                        n_hot_for_feature = sum(instance[self.one_hot_schema[feat_name]])
                        valid_i = valid_i and (n_hot_for_feature == 1)
                if self.col_types[col_id] == FeatureType.Discrete:
                    # discrete values must be one of the allowed values
                    valid_i = valid_i and (instance[col_id] in self.possible_vals[col_id])
            all_valid = all_valid and valid_i
        return all_valid

    def check_one_hot_validity(self, x: np.ndarray, verbose: bool = False) -> bool:
        '''
        Given an array `x` with one hot encoded features. Check that each one hot feature has exactly one column set

        `x`: a numpy array of shape `(n_instances, n_features)`
        `one_hot_schema`: a dictionary which maps `{"feature_name": [i, j, ... , k]}` where `feature_name` is the name of the feature which was one hot encoded and `[i, j, ... , k]` are the column indices for the resulting one hot columns
        '''
        if len(x.shape) == 1:
            data = x.reshape(1, x.shape[0])
        else:
            data = x

        all_valid = True
        for i in range(data.shape[0]):
            for cat_column_name, sub_col_idxs in self.one_hot_schema.items():
                n_set_colums = sum(data[i][sub_col_idxs])
                if (n_set_colums != 1):
                    all_valid = False
                    if verbose:
                        print("failed one-hot encoding\t\t{}\t{}\t{}".format(i,
                                                                             cat_column_name, sum(data[i][sub_col_idxs])))
        return all_valid

    def dict_to_point(self, point_dict: dict) -> np.ndarray:
        """
        Takes a dictionary of {"xi" : <number>, ...} and converts it to a numpy array
        """
        point = np.zeros(shape=(self.ncols,))
        for col, val in point_dict.items():
            point[self.col_to_idx[col]] = val
        return point

    def point_to_dict(self, point: np.ndarray) -> dict:
        """
        Takes a a numpy array and converts it to a dictionary of the form {"xi" : <number>, ...}
        """
        point_dict = dict()
        for i in range(point.shape[0]):
            point_dict["x{:d}".format(i)] = point[i]
        return point_dict

    def rect_to_dict(self, rect: np.ndarray) -> dict:
        """
        Takes a numpy array representing a rectangular counterfactual region and converts it to a dict
        """
        INFTY = 100000000000000
        rect_dict = dict()
        rect[rect == -np.inf] = -INFTY
        rect[rect == np.inf] = INFTY
        for i in range(self.ncols):
            rect_dict["x{:d}".format(i)] = [rect[i, LOWER], rect[i, UPPER]]
        return rect_dict


def rescale_numeric(x: np.ndarray, ds_info: DataInfo, scale_up=True):
    # if the datasets not normalized, we're dong
    if not ds_info.normalize_numeric:
        return x
    for i in range(ds_info.ncols):
        col_type = ds_info.col_types[i]
        if col_type == FeatureType.Numeric:
            min_val, max_val = ds_info.col_scales[i]
            if scale_up:
                x[:, i] * (max_val - min_val) + min_val
            else:  # scaling down
                x[:, i] = (x[:, i] - min_val) / (max_val - min_val)
    return x


def rescale_discrete(x: np.ndarray, ds_info: DataInfo, scale_up=True):
    # if the datasets not normalized, we're dong
    if not ds_info.normalize_discrete:
        return x
    for i in range(ds_info.ncols):
        col_type = ds_info.col_types[i]
        if col_type == FeatureType.Discrete:
            min_val, max_val = ds_info.col_scales[i]
            if scale_up:
                x[:, i] * (max_val - min_val) + min_val
            else:  # scaling down
                x[:, i] = (x[:, i] - min_val) / (max_val - min_val)
    return x


def load_data(dataset_name, normalize_numeric=True, normalize_discrete=True, do_convert=False) -> tuple[np.ndarray, np.ndarray, DataInfo]:
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
    elif dataset_name == "loans":
        x, y, ds_info = util_load_loans()
    else:
        print("ERROR NO SUCH DATASET")
        exit(0)
    # get the feature ranges for x
    ds_info.get_possible_vals(x, do_convert)
    # normalize numeric and discrete features to [0, 1]
    ds_info.normalize_data(x, normalize_numeric, normalize_discrete)
    return x, y, ds_info


def get_json_paths(dataset_name: str) -> tuple[str, str]:
    """
    Returns the paths the JSON files corresponding to the given dataset information

    Parameters
    ----------
    dataset_name : the short name of a dataset from DS_NAMES

    Returns
    -------
    json_paths : a tuple[ds_details_path, human_readable_path]
    """
    # check the dataset name is valid
    if dataset_name not in DS_NAMES:
        print("ERROR NO SUCH DATASET")
        exit(0)

    # get the paths of the JSON files
    data_directory = str(Path(DS_PATHS[dataset_name]).parent)
    ds_details_path = data_directory + "/dataset_details.json"
    human_readable_path = data_directory + "/human_readable.json"

    # return them as a tuple
    json_paths = [ds_details_path, human_readable_path]
    return json_paths


def type_to_enum(col_types) -> list[FeatureType]:
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


def action_to_enum(col_actionability) -> list[FeatureActionability]:
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


def one_hot_encode(input: np.ndarray, col_names: list[str], col_types: list[FeatureType]) -> DataInfo:
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

    # allowing alteration on all features
    one_hot_actions = [FeatureActionability.Free for _ in range(len(one_hot_names))]
    # create the data info object
    ds_info = DataInfo(one_hot_names, one_hot_types, one_hot_actions, one_hot_schema)
    return x, ds_info


def load_facet_data(ds_name: str) -> tuple[np.ndarray, np.ndarray, DataInfo]:
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
    col_names: list[str] = list(data.columns)
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

    x, ds_info = one_hot_encode(input, input_col_names, input_col_types)
    y = target.astype(int)

    return x, y, ds_info


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


def util_load_loans():
    data = pd.read_csv(DS_PATHS["loans"])
    colnames = list(data.columns)
    data = data.dropna().to_numpy()
    labels = data[:, -1:].squeeze()

    y = np.zeros(labels.shape[0], dtype=int)
    y[labels == "Y"] = 1
    x = data[:, :-1].astype(float)

    ds_info = DataInfo.generic(ncols=x.shape[1])
    ds_info.col_names = colnames[:-1]
    return x, y, ds_info
