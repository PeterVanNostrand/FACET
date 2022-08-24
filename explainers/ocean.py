import numpy as np
import pandas as pd
# from baselines.ocean.BuildCounterFactualSeekedSet import buildCounterFactualSeekedFile
from explainers.explainer import Explainer
from baselines.ocean.RandomForestCounterFactual import RandomForestCounterFactualMilp
from baselines.ocean.CounterFactualParameters import *
from baselines.ocean.RunExperimentsRoutines import trainModelAndSolveCounterFactuals, runNumericalExperiments

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manager import MethodManager


class OCEAN(Explainer):
    '''
    A wrapper for the Optimal Counterfactual ExplANations method developed in "ptimal Counterfactual Explanations in Tree Ensembles." Code was pulled from https://github.com/vidalt/OCEAN. The original paper can be found at https://arxiv.org/pdf/2106.06631.pdf
    '''

    def __init__(self, manager, hyperparameters=None):
        self.manager: MethodManager = manager
        self.parse_hyperparameters(hyperparameters)

    def parse_hyperparameters(self, hyperparameters: dict) -> None:
        params: dict = hyperparameters.get("OCEAN")
        if params.get("ocean_norm") is None:
            self.objectiveNorm = 2
            print("No ocean_norm set, using L2")
        else:
            self.objectiveNorm = hyperparameters.get("ocean_norm")

    def prepare(self, data=None):
        pass

    def explain(self, x, y):
        counterfactual_classes = ((y - 1) * -1)
        xprime = np.empty(shape=x.shape)
        xprime[:, :] = np.inf

        for i in range(x.shape[0]):
            to_explain = x[i].copy()
            sample = [pd.Series(to_explain, dtype=np.float64, name=str(i))]
            desired_label = counterfactual_classes[i]
            feat_index = []
            for j in range(x.shape[1]):
                feat_index.append("F{}".format(j))
            sample[0].index = feat_index

            feat_types = [FeatureType.Numeric for _ in range(x.shape[1])]
            possible_vals = [[] for _ in range(x.shape[1])]
            feat_actionability = [FeatureActionnability.Free for _ in range(x.shape[1])]

            randomForestMilp = RandomForestCounterFactualMilp(
                classifier=self.manager.random_forest.model,
                sample=sample,
                outputDesired=desired_label,
                isolationForest=None,
                constraintsType=TreeConstraintsType.LinearCombinationOfPlanes,
                objectiveNorm=2,
                mutuallyExclusivePlanesCutsActivated=True,
                strictCounterFactual=True,
                verbose=False,
                binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y,
                featuresActionnability=feat_actionability,
                featuresType=feat_types,
                featuresPossibleValues=possible_vals,
                randomCostsActivated=False
            )
            randomForestMilp.buildModel()
            randomForestMilp.solveModel()
            xprime[i] = np.array(randomForestMilp.x_sol[0])

        # check that all counterfactuals result in a different class
        preds = self.manager.predict(xprime)
        failed_explanation = (preds == y)
        xprime[failed_explanation] = np.tile(np.inf, x.shape[1])
        return xprime
