import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from baselines.mace.batchTest import generateExplanations
from baselines.mace.loadData import Dataset, DatasetAttribute, loadDataset
from baselines.ocean.CounterFactualParameters import FeatureActionability, FeatureType
from explainers.explainer import Explainer
from dataset import DataInfo
from dataset import rescale_discrete


if TYPE_CHECKING:
    from manager import MethodManager

MACE_TYPE_FROM_FACET_TYPE = {
    FeatureType.Numeric: "numeric-real",
    FeatureType.Binary: "binary",
    FeatureType.Discrete: "numeric-int",  # formerly numeric-int
    FeatureType.Categorical: "categorical",
    FeatureType.CategoricalNonOneHot: "ordinal",  # ! WARNING NOT MACE SUPPORTED
}

MACE_ACTION_FROM_FACET_ACTION = {
    FeatureActionability.Free: "any",
    FeatureActionability.Fixed: "none",
    FeatureActionability.Increasing: "same-or-increase",
    FeatureActionability.Predict: "none",
}

MACE_MUTABILITY_FROM_FACET_ACTION = {
    FeatureActionability.Free: True,
    FeatureActionability.Fixed: False,
    FeatureActionability.Increasing: True,
    FeatureActionability.Predict: False,
}


class MACE(Explainer):
    '''
    A wrapper for the Model Agnostic Counterfactual Explanations method developed in "Model-Agnostic Counterfactual Explanations for Consequential Decisions." Code was pulled from https://github.com/amirhk/mace. The original paper can be found at https://proceedings.mlr.press/v108/karimi20a/karimi20a.pdf
    '''
    # TODO: Apply attribute typing and one-hot encoding (convert from FACET/OCEAN schema)

    def __init__(self, manager, hyperparameters=None):
        self.manager: MethodManager = manager
        self.parse_hyperparameters(hyperparameters)

    def parse_hyperparameters(self, hyperparameters: dict) -> None:
        self.hyperparameters = hyperparameters

        params = hyperparameters.get("MACE")

        verbose = params.get("mace_verbose")
        if verbose is None:
            self.verbose = False
        else:
            self.verbose = verbose

        # maximum time to attempt to explain a sample
        maxtime = params.get("mace_maxtime")
        if maxtime is None:
            if self.verbose:
                print("No mace_maxtime provided, using None")
            self.maxtime = None
        else:
            self.maxtime = maxtime

        # epsilon
        epsilon = params.get("mace_epsilon")
        if epsilon is None:
            print("No mace_epsilon provided, using 1e-7")
            self.epsilon = 1e-7
        else:
            self.epsilon = epsilon

    def prepare(self, xtrain=None, ytrain=None):
        pass

    def get_mace_attributes(self) -> dict[str, DatasetAttribute]:
        '''
        Takes FACET/OCEAN encoded feature types and converts them to MACE DatasetAttributes
        '''
        attributes_non_hot = {}
        self.ds_info.mace_names_long = []
        self.ds_info.mace_names_kurz = []
        for i in range(self.ds_info.ncols):  # for each column
            # if i not in self.ds_info.reverse_one_hot_schema:  # if its not part of the one-hot encoding
            col_name = self.ds_info.col_names[i]

            # lower_bound, upper_bound = self.ds_info.col_scales[i]
            lower_bound = self.ds_info.possible_vals[i][0]
            upper_bound = self.ds_info.possible_vals[i][-1]

            # if this column in one hot encoded
            if i in self.ds_info.reverse_one_hot_schema:
                parent_name_long = self.ds_info.reverse_one_hot_schema[i]
                parent_name_kurz = self.ds_info.reverse_one_hot_schema[i]
                attr_type = "sub-categorical"
                # mace names one-hot columns as FeatName_cat_X, where X is the option
                sub_cat_cols = self.ds_info.one_hot_schema[parent_name_long]
                attr_name_long = "{}_cat_{}".format(parent_name_long, sub_cat_cols.index(i))
                # attr_name_kurz = "x{}".format(i)
                attr_name_kurz = "x{}_cat_{}".format(sub_cat_cols[0], sub_cat_cols.index(i))
            else:
                parent_name_long = -1
                parent_name_kurz = -1
                attr_type = MACE_TYPE_FROM_FACET_TYPE[self.ds_info.col_types[i]]
                attr_name_long = col_name
                attr_name_kurz = "x{}".format(i)

            self.ds_info.mace_names_long.append(attr_name_long)
            self.ds_info.mace_names_kurz.append(attr_name_kurz)

            attributes_non_hot[attr_name_long] = DatasetAttribute(
                attr_name_long=attr_name_long,
                attr_name_kurz=attr_name_kurz,
                attr_type=attr_type,
                node_type="input",
                actionability=MACE_ACTION_FROM_FACET_ACTION[self.ds_info.col_actions[i]],
                mutability=MACE_MUTABILITY_FROM_FACET_ACTION[self.ds_info.col_actions[i]],
                parent_name_long=parent_name_long,
                parent_name_kurz=parent_name_kurz,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        # add the output column
        attributes_non_hot["y"] = DatasetAttribute(
            attr_name_long="label",
            attr_name_kurz='y',
            attr_type='binary',
            node_type='output',
            actionability='none',
            mutability=False,
            parent_name_long=-1,
            parent_name_kurz=-1,
            lower_bound=0.0,
            upper_bound=1.0)
        return attributes_non_hot

    def prepare_dataset(self, x: np.ndarray, y: np.ndarray, ds_info: DataInfo) -> None:
        self.ds_info = ds_info.copy()
        attributes = self.get_mace_attributes()

        # MACE requires numeric-int features to be unscaled while FACET and OCEAN used the scaled versions
        # here we unscale the discrete values, we'll also have to scale/rescale elsewhere in the MACE code
        x_unscaled = rescale_discrete(x.copy(), self.ds_info)
        df = pd.DataFrame(x_unscaled)
        df.columns = self.ds_info.mace_names_long
        y_pred = self.manager.model.predict(x)
        df["label"] = y_pred
        df = df + 0  # convert boolean values to numeric
        df = df.reset_index(drop=True)
        df = df.dropna()
        df = df.astype('float64')

        self.dataset_obj = Dataset(data_frame=df, attributes=attributes,
                                   is_one_hot=self.ds_info.one_hot_schema, dataset_name="dataset")

    def explain(self, x: np.ndarray, y: np.ndarray, k: int = 1, constraints: np.ndarray = None, weights: np.ndarray = None, max_dist: float = np.inf, opt_robust: bool = False, min_robust: float = None) -> np.ndarray:
        xprime = []

        approach_string = "MACE_eps_{}".format(self.epsilon)
        explanation_file_name = "mace_temp.log"
        norm_type_string = "two_norm"
        rf_model = self.manager.model.model

        mace_col_names = list(self.dataset_obj.data_frame_kurz.columns)
        progress = tqdm(total=x.shape[0], desc="MACE", leave=False)
        for i in range(x.shape[0]):  # for each instance
            factual_sample = {}

            for j in range(self.ds_info.ncols):
                factual_sample[mace_col_names[j]] = x[i][j]

            factual_sample['y'] = bool(y[i])
            # explain the sample
            explanation = doMACEExplanationWithMaxTime(
                self.maxtime,
                approach_string,
                explanation_file_name,
                rf_model,
                self.dataset_obj,
                factual_sample,
                norm_type_string,
                self.verbose
            )
            # get MACE's explanation
            exp = [np.inf for _ in range(x.shape[1])]
            if explanation is not None and len(explanation["cfe_sample"]) != 0:
                for j in range(len(factual_sample)-1):
                    attr_name_kurz = mace_col_names[j]
                    exp[j] = explanation["cfe_sample"][attr_name_kurz]
            else:
                print("mace returned no explanation. time out?")
            xprime.append(exp)
            progress.update()
        progress.close()

        # temporarily replace any explanations which timed out with zeros for prediction
        xprime = np.array(xprime)
        idx_inf = (xprime == np.inf).any(axis=1)
        xprime[idx_inf] = np.tile(0, (x.shape[1],))
        # if MACE finds a solution of the same class, remove it
        y_pred = self.manager.predict(xprime)
        idx_failed_explanation = (y_pred == y)
        # replace bad explanations and timed out explanations with [inf inf ... inf]
        xprime[idx_failed_explanation] = np.tile(np.inf, (x.shape[1],))
        xprime[idx_inf] = np.tile(np.inf, (x.shape[1],))

        return xprime


def doMACEExplanationWithMaxTime(maxTime,
                                 approach_string,
                                 explanation_file_name,
                                 model_trained,
                                 dataset_obj,
                                 factual_sample,
                                 norm_type_string,
                                 verbose=False):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=doMACEExplanationWithQueueCatch, args=(
        q,
        approach_string,
        explanation_file_name,
        model_trained,
        dataset_obj,
        factual_sample,
        norm_type_string)
    )
    p.start()
    p.join(maxTime)
    if p.is_alive():
        p.terminate()
        if verbose:
            print("killing after", maxTime, "second")
        return None
    else:
        return q.get()


def doMACEExplanationWithQueueCatch(queue,
                                    approach_string,
                                    explanation_file_name,
                                    model_trained,
                                    dataset_obj,
                                    factual_sample,
                                    norm_type_string
                                    ):
    try:
        doMACEExplanationWithQueue(queue,
                                   approach_string,
                                   explanation_file_name,
                                   model_trained,
                                   dataset_obj,
                                   factual_sample,
                                   norm_type_string
                                   )
    except (AssertionError):
        print("solver returned error for", approach_string)
        queue.put(None)


def doMACEExplanationWithQueue(queue,
                               approach_string,
                               explanation_file_name,
                               model_trained,
                               dataset_obj,
                               factual_sample,
                               norm_type_string
                               ):
    queue.put(doMACEExplanation(
        approach_string,
        explanation_file_name,
        model_trained,
        dataset_obj,
        factual_sample,
        norm_type_string
    ))


def doMACEExplanation(approach_string,
                      explanation_file_name,
                      model_trained,
                      dataset_obj,
                      factual_sample,
                      norm_type_string):
    return generateExplanations(
        approach_string=approach_string,
        explanation_file_name=explanation_file_name,
        model_trained=model_trained,
        dataset_obj=dataset_obj,
        factual_sample=factual_sample,
        norm_type_string=norm_type_string
    )


def getEpsilonInString(approach_string):
    tmp_index = approach_string.find('eps')
    epsilon_string = approach_string[tmp_index + 4: tmp_index + 8]
    return float(epsilon_string)
