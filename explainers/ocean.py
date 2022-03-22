import numpy as np
import pandas as pd
from explainers.explainer import Explainer
from baselines.ocean.RandomForestCounterFactual import RandomForestCounterFactualMilp
from baselines.ocean.CounterFactualParameters import *


class OCEAN(Explainer):
    def __init__(self, model, hyperparameters=None):
        self.model = model
        self.hyperparameters = hyperparameters

        if hyperparameters.get("ocean_norm") is None:
            self.objectiveNorm = 2
            print("No ocean_norm set, using L2")
        else:
            self.objectiveNorm = hyperparameters.get("ocean_norm")

    def prepare(self, data=None):
        pass

    def explain(self, x, y):
        xprime = x.copy()  # an array for the constructed contrastive examples

        for i in range(len(x)):  # sample, label in zip(x, y):
            sample = [pd.Series(x[i])]
            label = y[i]
            if label == 1:
                desired_label = 0
            else:
                desired_label = 1

            featuresType = [FeatureType.Numeric] * x.shape[1]
            featuresPossibleValues = [[]] * x.shape[1]

            randomForestMilp = RandomForestCounterFactualMilp(
                self.model.detectors[0].model, sample, desired_label, objectiveNorm=self.objectiveNorm, binaryDecisionVariables=BinaryDecisionVariables.PathFlow_y, constraintsType=TreeConstraintsType.LinearCombinationOfPlanes, randomCostsActivated=False, featuresType=featuresType, featuresPossibleValues=featuresPossibleValues, mutuallyExclusivePlanesCutsActivated=True, strictCounterFactual=True)  # , binaryDecisionVariables=BinaryDecisionVariables.LeftRight_lambda, constraintsType=TreeConstraintsType.BigM, , strictCounterFactual=True , )
            randomForestMilp.buildModel()
            randomForestMilp.solveModel()
            xprime[i] = np.array(randomForestMilp.x_sol[0])

        # As OCEAN is an appoximation method it is possible that its found solution doesn't result in a class change
        # remove examples for which this is the case
        y_pred = self.model.predict(xprime)
        idx_failed_explanation = (y_pred == y)
        xprime[idx_failed_explanation] = np.tile(np.inf, (x.shape[1],))

        return xprime
