import numpy as np
import pandas as pd
from explainers.explainer import Explainer

from baselines.rfocse.datasets import DatasetInfoBuilder
from baselines.rfocse.rfocse_main import batch_extraction
from baselines.rfocse.rfocse_main import CounterfactualSetExplanation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heead import HEEAD


class RFOCSE(Explainer):
    def __init__(self, model, hyperparameters=None):
        self.model: HEEAD = model
        self.hyperparameters = hyperparameters

    def prepare(self, data=None):
        self.data = data

    def explain(self, x: np.ndarray, y):
        # an array for the constructed contrastive examples
        xprime = np.empty(shape=x.shape)
        xprime[:, :] = np.inf

        # build the RF-OCSE dataset info file, contains details on feature type and range
        ds_builder: DatasetInfoBuilder = DatasetInfoBuilder(name="none")
        attr_names = []
        for i in range(x.shape[1]):
            name = "x{:d}".format(i)
            attr_names.append(name)
            ds_builder.add_numerical_variable(position=i, lower_bound=-100, upper_bound=100, name=name)
        ds_info = ds_builder.create_dataset_info()

        rf_model = self.model.detectors[0].model

        results = batch_extraction(sklearn_rf=rf_model, dataset_info=ds_info, X=x, max_distance=100, log_every=-
                                   1, max_iterations=20_000_000, export_rules_file=None, dataset=self.data)

        for idx, res in zip(range(x.shape[0]), results):
            if 'explanation' in res:
                cs: CounterfactualSetExplanation = res.pop('explanation')
                xprime[idx] = cs.sample_counterfactual(x[idx], epsilon=0.01)
            else:
                print("no explanation")

        idx_no_examples = (xprime == np.inf).any(axis=1)
        xprime[idx_no_examples] = np.tile(0, x.shape[1])
        y_pred = self.model.predict(xprime)
        idx_failed_explanation = (y_pred == y)

        xprime[idx_failed_explanation] = np.tile(np.inf, (x.shape[1],))
        xprime[idx_no_examples] = np.tile(np.inf, (x.shape[1],))

        return xprime
