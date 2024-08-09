from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tqdm.auto import tqdm

from baselines.ocean.CounterFactualParameters import BinaryDecisionVariables, TreeConstraintsType
from baselines.ocean.RandomForestCounterFactual import RandomForestCounterFactualMilp
# from baselines.ocean.BuildCounterFactualSeekedSet import buildCounterFactualSeekedFile
from explainers.explainer import Explainer

if TYPE_CHECKING:
    from dataset import DataInfo
    from manager import MethodManager


class OCEAN(Explainer):
    '''
    A wrapper for the Optimal Counterfactual ExplANations method developed in "Optimal Counterfactual Explanations in Tree Ensembles." Code was pulled from https://github.com/vidalt/OCEAN. The original paper can be found at https://arxiv.org/pdf/2106.06631.pdf
    '''

    def __init__(self, manager, hyperparameters=None):
        self.manager: MethodManager = manager
        self.parse_hyperparameters(hyperparameters)

    def parse_hyperparameters(self, hyperparameters: dict) -> None:
        params: dict = hyperparameters.get("OCEAN")
        # ocean norm
        if params.get("ocean_norm") is None:
            self.objectiveNorm = 2
            print("No ocean_norm set, using L2")
        else:
            self.objectiveNorm = hyperparameters.get("ocean_norm")

        # isolation forest
        if params.get("ocean_ilf") is None:
            self.use_ilf = False
            print("No ocean_ilf set, defaulting to False")
        else:
            self.use_ilf = params.get("ocean_ilf")

    def prepare(self, xtrain: np.ndarray = None, ytrain: np.ndarray = None):
        rf_nclasses = self.manager.model.model.n_classes_
        if self.use_ilf:
            self.ilfs = []
            for i in range(rf_nclasses):
                # create the isolation forest model
                ilf_max_samples = 32
                ilf_n_estimators = 100
                ilf = IsolationForest(random_state=self.manager.random_state,
                                      max_samples=ilf_max_samples,
                                      n_estimators=ilf_n_estimators, contamination=0.1)
                idx_match = (ytrain == 1)
                data = xtrain[idx_match]
                ilf.fit(data)
                self.ilfs.append(ilf)
        else:
            self.ilfs = [None for _ in range(rf_nclasses)]

    def prepare_dataset(self, x: np.ndarray, y: np.ndarray, ds_info) -> None:
        self.ds_info: DataInfo = ds_info

    def explain(self, x: np.ndarray, y: np.ndarray, k: int = 1, constraints: np.ndarray = None, weights: np.ndarray = None, max_dist: float = np.inf, opt_robust: bool = False, min_robust: float = None, return_regions: bool = False) -> np.ndarray:
        counterfactual_classes = ((y - 1) * -1)
        xprime = np.empty(shape=x.shape)
        xprime[:, :] = np.inf

        progress = tqdm(total=x.shape[0], desc="OCEAN", leave=False)
        for i in range(x.shape[0]):
            to_explain = x[i].copy()
            sample = [pd.Series(to_explain, dtype=np.float64, name=str(i))]
            desired_label = counterfactual_classes[i]
            feat_index = []
            for j in range(1, x.shape[1]+1):
                feat_index.append("F{}".format(j))
            sample[0].index = feat_index

            randomForestMilp = RandomForestCounterFactualMilp(
                classifier=self.manager.model.model,
                sample=sample,
                outputDesired=desired_label,
                isolationForest=self.ilfs[desired_label],
                constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                objectiveNorm=2,
                mutuallyExclusivePlanesCutsActivated=True,
                strictCounterFactual=True,
                verbose=False,
                featuresType=self.ds_info.col_types,
                featuresPossibleValues=self.ds_info.possible_vals,
                featuresActionnability=self.ds_info.col_actions,
                oneHotEncoding=self.ds_info.one_hot_schema,
                binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
                randomCostsActivated=False
            )

            randomForestMilp.buildModel()
            randomForestMilp.solveModel()
            xprime[i] = np.array(randomForestMilp.x_sol[0])
            progress.update()
        progress.close()

        # check that all counterfactuals result in a different class
        preds = self.manager.predict(xprime)
        failed_explanation = (preds == y)
        xprime[failed_explanation] = np.tile(np.inf, x.shape[1])
        xprime += 0  # make -0.0 values 0.0

        return xprime
