import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from baselines.mace.batchTest import generateExplanations
from baselines.mace.loadData import Dataset, DatasetAttribute, loadDataset
from baselines.ocean.CounterFactualParameters import FeatureActionability, FeatureType
from explainers.explainer import Explainer

if TYPE_CHECKING:
    from manager import MethodManager

MACE_TYPE_FROM_FACET_TYPE = {
    FeatureType.Numeric: "numeric-real",
    FeatureType.Binary: "binary",
    FeatureType.Discrete: "numeric-int",
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
# prepare the MACE dataset info object
# non_hot_df = data.iloc[2:].astype(np.float64)
# attributes_non_hot = get_mace_non_hot(data=non_hot_df, col_names=col_names, col_types=col_types, col_actions=col_actionabiltiy)


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
                attr_name_kurz=f"x{i}" if col_actions[i] != FeatureActionability.Predict else "y",
                attr_type=MACE_TYPE_FROM_FACET_TYPE[col_types[i]],
                node_type="input" if col_actions[i] != FeatureActionability.Predict else "output",
                actionability=MACE_ACTION_FROM_FACET_ACTION[col_actions[i]],
                mutability=MACE_MUTABILITY_FROM_FACET_ACTION[col_actions[i]],
                parent_name_long=-1,
                parent_name_kurz=-1,
                lower_bound=data[col_name].min(),
                upper_bound=data[col_name].max()
            )

    return attributes_non_hot


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

    def prepare_dataset(self, x: np.ndarray, y: np.ndarray, ds_info) -> None:
        df = pd.DataFrame(x)
        col_names = []
        for i in range(x.shape[1]):
            col_names.append("x" + str(i))
        df.columns = col_names
        y_pred = self.manager.random_forest.predict(x)
        df.insert(0, "alpha (label)", y_pred)
        df = df + 0  # convert boolean values to numeric
        df = df.reset_index(drop=True)
        df = df.dropna()
        df = df.astype('float64')

        dataset_obj: Dataset = loadDataset(dataset_name="cust_data", return_one_hot=False,
                                           load_from_cache=False, debug_flag=False, my_df=df.copy())
        self.dataset_obj = dataset_obj

    def explain(self, x: np.ndarray, y: np.ndarray, k: int = 1, constraints: np.ndarray = None, weights: np.ndarray = None, max_dist: float = np.inf, opt_robust: bool = False, min_robust: float = None) -> np.ndarray:
        xprime = []

        approach_string = "MACE_eps_{}".format(self.epsilon)
        explanation_file_name = "mace_temp.log"
        norm_type_string = "two_norm"
        rf_model = self.manager.random_forest.model

        progress = tqdm(total=x.shape[0], desc="MACE", leave=False)
        for i in range(x.shape[0]):
            factual_sample = {}
            for j in range(x.shape[1]):
                factual_sample["x{}".format(j)] = x[i][j]
            factual_sample['y'] = bool(y[i])

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
            # explanation = generateExplanations(
            #     approach_string=approach_string,
            #     explanation_file_name=explanation_file_name,
            #     model_trained=rf_model,
            #     dataset_obj=self.dataset_obj,
            #     factual_sample=factual_sample,
            #     norm_type_string=norm_type_string
            # )
            exp = [np.inf for _ in range(x.shape[1])]
            if explanation is not None:
                for i in range(len(factual_sample)-1):
                    exp[i] = explanation["cfe_sample"]["x{}".format(i)]
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
