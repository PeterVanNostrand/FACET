import numpy as np
import pandas as pd
from explainers.explainer import Explainer

from baselines.rfocse.datasets import DatasetInfoBuilder
from baselines.rfocse.rfocse_main import batch_extraction
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
        xprime = x.copy()  # an array for the constructed contrastive examples

        # convert the dataset into a RF-OCSE compatible format

        # build the RF-OCSE dataset info file, contains details on feature type and range
        ds_builder: DatasetInfoBuilder = DatasetInfoBuilder(name="none")
        attr_names = []
        for i in range(x.shape[1]):
            name = "x{:d}".format(i)
            attr_names.append(name)
            ds_builder.add_numerical_variable(position=i, lower_bound=0.0, upper_bound=1.0, name=name)
        ds_info = ds_builder.create_dataset_info()

        rf_model = self.model.detectors[0].model

        results = batch_extraction(sklearn_rf=rf_model, dataset_info=ds_info, X=x, max_distance=100, log_every=-
                                   1, max_iterations=20_000_000, export_rules_file=None, dataset=self.data)

        for res in results:
            if 'explanation' in res:
                cs = res.pop('explanation')
                res['feature_bounds'] = cs.feat_conditions
                res['feature_bounds_compact'] = res.pop('explanation_compact').feat_conditions
                res['feature_bounds_compact_count'] = res.pop('explanation_compact_count').feat_conditions
                res['counterfactual_sample'] = cs.sample_counterfactual(X_test.loc[idx], epsilon=0.005)
                res['factual_class'] = rf.predict(np.array(X_test.loc[idx]).reshape(1, -1))[0]
                res['counterfactual_distance'] = res['distance']
                res['counterfactual_class'] = rf.predict(np.array(res['counterfactual_sample']).reshape(1, -1))[0]
                res['valid_'] = res['factual_class'] != res['counterfactual_class']

                # res['distance'] = getDistanceBetweenSamples(
                #     to_mace_sample(cs.sample_counterfactual(X_test.loc[idx], epsilon=0.00005), dataset_obj_mace, colnames),
                #     to_mace_sample(X_test.loc[idx], dataset_obj_mace, colnames),
                #     'one_norm',
                #     dataset_obj_mace
                # )

        # Get the model and prepare data
        # df = df.drop(['label'], axis=1)
        # model_trained = self.model.detectors[0].model
        # df['y'] = model_trained.predict(x)
        # iterate_over_data_dict = df.T.to_dict()

        # parameters

        # explanation = genExp(
        #     explanation_file_name=explanation_file_name,
        #     model_trained=self.model.detectors[0].model,
        #     dataset_obj=
        # )

        # explanation_object = generateExplanationsWithMaxTime(
        #     self.maxtime,
        #     approach_string,
        #     explanation_file_name,
        #     model_trained,
        #     dataset_obj,
        #     factual_sample,
        #     norm_type_string,
        #     observable_data_dict,  # used solely for minimum_observable method
        #     standard_deviations,  # used solely for feature_tweaking method
        # )
        # example = explanation_object['counterfactual_sample']
        # if example:
        #     for i in range(len(col_names)):
        #         xprime[factual_sample_index, i] = example[col_names[i]]
        # else:
        #     # no example found
        #     xprime[factual_sample_index] = np.tile(np.inf, (x.shape[1],))
        # print(explained_samples)
        # explained_samples += 1

        # if MACE finds a solution of the same class, remove it
        # to do so we need to predict, and so temporarily swap the infs for zeros
        idx_no_examples = (xprime == np.inf).any(axis=1)
        xprime[idx_no_examples] = np.tile(0, x.shape[1])
        y_pred = self.model.predict(xprime)
        idx_failed_explanation = (y_pred == y)

        xprime[idx_failed_explanation] = np.tile(np.inf, (x.shape[1],))
        xprime[idx_no_examples] = np.tile(np.inf, (x.shape[1],))

        return xprime
