import numpy as np
import pandas as pd
from explainers.explainer import Explainer

# from baselines.mace.newLoadData import loadDataset
# from baselines.mace.newBatchTest import generateExplanationsWithMaxTime
# from baselines.mace.newBatchTest import generateExplanations
from baselines.mace.generateSATExplanations import genExp
from baselines.mace.loadData import loadDataset
from baselines.mace.loadData import Dataset
from sklearn.model_selection import train_test_split
from baselines.mace.batchTest import generateExplanations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manager import MethodManager


class MACE(Explainer):
    def __init__(self, manager, hyperparameters=None):
        self.manager: MethodManager = manager
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

        # epsilon
        epsilon = params.get("mace_epsilon")
        if maxtime is None:
            print("No mace_epsilon provided, using 60")
            self.epsilon = 1e-3
        else:
            self.epsilon = epsilon

    def prepare(self, data=None):
        pass

    def prepare_dataset(self, x, y):
        df = pd.DataFrame(x)
        col_names = []
        for i in range(x.shape[1]):
            col_names.append("x" + str(i))
        # col_names = ["bravo", "charlie", "delta", "echo",
        #              "foxtrot", "golf", "hotel", "india", "juliett", "kilo"]
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

    def explain(self, x: np.ndarray, y: np.ndarray):
        # df = pd.DataFrame(x)
        # col_names = []
        # for i in range(x.shape[1]):
        #     col_names.append("x" + str(i))
        # df.columns = col_names
        # df.insert(0, "label", y)
        # df = df.astype('float64')
        # df = df.reset_index(drop=True)
        # df = df.dropna()

        # dataset_obj: Dataset = loadDataset(dataset_name="dataset", return_one_hot=False,
        #                                    load_from_cache=False, debug_flag=False, my_df=df)
        # df_kurz: pd.DataFrame = dataset_obj.data_frame_kurz
        # df_y = df_kurz["y"]
        # df_x = df_kurz[col_names]
        # X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=.7, random_state=0)
        xprime = []

        approach_string = "MACE_eps_{}".format(self.epsilon)
        explanation_file_name = "test.log"
        norm_type_string = "two_norm"
        rf_model = self.manager.random_forest.model

        # X_train, X_test, y_train, y_test = self.dataset_obj.getTrainTestSplit()
        # self.manager.random_forest.model.fit(X_train, y_train)
        # X_test_pred_labels = self.manager.random_forest.model.predict(X_test)
        # all_pred_data_df = X_test
        # all_pred_data_df['y'] = (X_test_pred_labels == 1.0)
        # iterate_over_data_df = all_pred_data_df
        # iterate_over_data_dict = iterate_over_data_df.T.to_dict()
        # iterate_over_data_dict = iterate_over_data_df.T.to_dict()
        # for factual_sample_index, factual_sample in iterate_over_data_dict.items():

        n_explained = 0
        for i in range(x.shape[0]):
            factual_sample = {}
            for j in range(x.shape[1]):
                factual_sample["x{}".format(j)] = x[i][j]
            factual_sample['y'] = bool(y[i])

            print("n_explained: {}".format(n_explained))
            explanation = generateExplanations(
                approach_string=approach_string,
                explanation_file_name=explanation_file_name,
                model_trained=rf_model,
                dataset_obj=self.dataset_obj,
                factual_sample=factual_sample,
                norm_type_string=norm_type_string
            )
            exp = [np.inf for _ in range(x.shape[1])]
            for i in range(len(factual_sample)-1):
                exp[i] = explanation["cfe_sample"]["x{}".format(i)]
            xprime.append(exp)

        xprime = np.array(xprime)

        # if MACE finds a solution of the same class, remove it
        # to do so we need to predict, and so temporarily swap the infs for zeros
        # idx_no_examples = (xprime == np.inf).any(axis=1)
        # xprime[idx_no_examples] = np.tile(0, x.shape[1])
        # y_pred = self.manager.predict(xprime)
        # idx_failed_explanation = (y_pred == y)

        # xprime[idx_failed_explanation] = np.tile(np.inf, (x.shape[1],))
        # xprime[idx_no_examples] = np.tile(np.inf, (x.shape[1],))

        # return xprime
        return None


def getEpsilonInString(approach_string):
    tmp_index = approach_string.find('eps')
    epsilon_string = approach_string[tmp_index + 4: tmp_index + 8]
    return float(epsilon_string)
