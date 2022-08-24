import numpy as np
import pandas as pd
from explainers.explainer import Explainer

from baselines.mace.newLoadData import loadDataset
from baselines.mace.newBatchTest import generateExplanationsWithMaxTime
from baselines.mace.newBatchTest import generateExplanations
from baselines.mace.generateSATExplanations import genExp


class MACE(Explainer):
    def __init__(self, manager, hyperparameters=None):
        self.manager = manager
        self.parse_hyperparameters(hyperparameters)

    def parse_hyperparameters(self, hyperparameters: dict) -> None:
        self.hyperparameters = hyperparameters

        params = hyperparameters.get("MACE")

        # threshold offest for picking new values
        maxtime = params.get("mace_maxtime")
        if maxtime is None:
            print("No mace_maxtime provided, using 60")
            self.maxtime = 60
        else:
            self.maxtime = maxtime

    def prepare(self, data=None):
        pass

    def explain(self, x, y):
        xprime = x.copy()  # an array for the constructed contrastive examples

        # MACE requires a dataframe object, lets build one
        df = pd.DataFrame(x)
        col_names = []
        for i in range(x.shape[1]):
            col_names.append("x" + str(i))
        df.columns = col_names
        df["label"] = y
        dataset_obj = loadDataset(dataset_name="dataset", data_frame_non_hot=df, return_one_hot=False)

        # Get the model and prepare data
        df = df.drop(['label'], axis=1)
        model_trained = self.manager.detectors[0].model
        df['y'] = model_trained.predict(x)
        iterate_over_data_dict = df.T.to_dict()

        # parameters
        approach_string = "MACE_eps_1e-3"
        explanation_file_name = "test.log"
        norm_type_string = "two_norm"
        observable_data_dict = iterate_over_data_dict
        standard_deviations = list(df.std())

        explained_samples = 1
        for factual_sample_index, factual_sample in iterate_over_data_dict.items():
            # note y is the predicted label not the true label
            factual_sample['y'] = bool(factual_sample['y'])
            explanation_object = generateExplanations(
                approach_string=approach_string,
                explanation_file_name=explanation_file_name,
                model_trained=model_trained,
                dataset_obj=dataset_obj,
                factual_sample=factual_sample,
                norm_type_string=norm_type_string,
                standard_deviations=standard_deviations,
                observable_data_dict=observable_data_dict
            )

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
            example = explanation_object['counterfactual_sample']
            if example:
                for i in range(len(col_names)):
                    xprime[factual_sample_index, i] = example[col_names[i]]
            else:
                # no example found
                xprime[factual_sample_index] = np.tile(np.inf, (x.shape[1],))
            print(explained_samples)
            explained_samples += 1

        # if MACE finds a solution of the same class, remove it
        # to do so we need to predict, and so temporarily swap the infs for zeros
        idx_no_examples = (xprime == np.inf).any(axis=1)
        xprime[idx_no_examples] = np.tile(0, x.shape[1])
        y_pred = self.manager.predict(xprime)
        idx_failed_explanation = (y_pred == y)

        xprime[idx_failed_explanation] = np.tile(np.inf, (x.shape[1],))
        xprime[idx_no_examples] = np.tile(np.inf, (x.shape[1],))

        return xprime
