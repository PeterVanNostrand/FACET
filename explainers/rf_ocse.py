import numpy as np
import pandas as pd
from explainers.explainer import Explainer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import multiprocessing as mp


# from fastcrf.rfocse_main import batch_extraction, CounterfactualSetExplanation
# from fastcrf.datasets import DatasetInfo, DatasetInfoBuilder

from baselines.rfocse.rfocse_main import batch_extraction, extract_single_counterfactual_set
from baselines.rfocse.extraction_problem import make_problem
from baselines.rfocse.converter import convert_rf_format
from baselines.rfocse.datasets import DatasetInfo, DatasetInfoBuilder

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manager import MethodManager
    from dataset import DataInfo


class RFOCSE(Explainer):
    '''
    An wapper of the Random Forest Open Counterfactual Sets Method developed in the paper "Random forest explanability using counterfactual sets". This version has been converted from the released Cython implementation located at https://github.com/rrunix/libfastcrf/ which runs compiled to Python which is interpreted at runtime. This allows for a fair comparison between methods and removes that implementations requirement to run on Linux. The source paper can be found at https://www.sciencedirect.com/science/article/abs/pii/S1566253520303134
    '''

    def __init__(self, manager, hyperparameters=None):
        self.manager: MethodManager = manager
        self.parse_hyperparameters(hyperparameters)

    def parse_hyperparameters(self, hyperparameters: dict) -> None:
        self.hyperparameters = hyperparameters
        params: dict = hyperparameters.get("RFOCSE")
        if params.get("rfoce_transform") is None:
            self.perform_transform = True
            print("No rfoce_transform provided, using True")
        else:
            self.perform_transform = params.get("rfoce_transform")

        if params.get("rfoce_offset") is None:
            self.offset = 0.001
            print("No rfoce_offset provided, using 0.001")
        else:
            self.offset = params.get("rfoce_offset")

        verbose = params.get("rfocse_verbose")
        if verbose is None:
            self.verbose = False
        else:
            self.verbose = verbose

        # maximum time to attempt to explain a sample
        maxtime = params.get("rfoce_maxtime")
        if maxtime is None:
            if self.verbose:
                print("No rfoce_maxtime provided, using None")
            self.maxtime = None
        else:
            self.maxtime = maxtime

    def prepare_dataset(self, x: np.ndarray, y: np.ndarray, ds_info) -> None:
        self.ds_info: DataInfo = ds_info

    def prepare(self, xtrain=None, ytrain=None):
        data = xtrain
        self.data = data
        df = pd.DataFrame(data)
        y = self.manager.predict(data)
        df["y"] = y
        override_dtypes = {}
        col_names = []
        for i in range(data.shape[1]):
            name = "x{}".format(i)
            col_names.append(name)
            override_dtypes[name] = float
        col_names.append("y")
        self.col_names = col_names
        df.columns = col_names

        dataset_info, X, y = self.process_pandas_dataset(df[list(override_dtypes.keys()) + ['y']], 'y',
                                                         **self.get_dataset_args(dict(override_feature_types=override_dtypes), {}))
        self.dataset_info: DatasetInfo = dataset_info
        self.Xtrain: pd.DataFrame = X

    def explain(self, x: np.ndarray, y: np.ndarray, k: int = 1, constraints: np.ndarray = None, weights: np.ndarray = None, max_dist: float = np.inf, opt_robust: bool = False, min_robust: float = None) -> np.ndarray:
        if self.perform_transform:
            x = self.float_transformer.transform(x)
        # an array for the constructed contrastive examples
        X_test = pd.DataFrame(x)
        X_test.columns = self.col_names[:-1]
        xprime = np.empty(shape=x.shape)
        xprime[:, :] = np.inf
        dataset_info: DatasetInfo = self.dataset_info
        sklearn_rf = self.manager.random_forest.model
        X_train = self.Xtrain

        # Make the RFOCSE problem
        search_closest = True
        log_every = -1
        max_iterations = 20_000_000
        dataset = X_train.values
        parsed_rf = convert_rf_format(sklearn_rf, dataset_info)
        problem = make_problem(parsed_rf, dataset_info, search_closest, log_every, max_iterations)

        # explain each instance
        for idx, observation in zip(X_test.index, X_test.values):
            explanation = doRFOCSEExplanationWithMaxTime(self.maxtime,
                                                         problem,
                                                         sklearn_rf,
                                                         parsed_rf,
                                                         dataset_info,
                                                         observation,
                                                         dataset,
                                                         self.offset,
                                                         verbose=self.verbose)
            if explanation is not None:
                xprime[idx] = explanation

        # locate non-counterfactual examples and replace them with [np.inf, ... , np.inf]
        idx_no_examples = (xprime == np.inf).any(axis=1)
        xprime[idx_no_examples] = np.tile(0, x.shape[1])
        y_pred = self.manager.predict(xprime)
        idx_failed_explanation = (y_pred == y)
        xprime[idx_failed_explanation] = np.tile(np.inf, (x.shape[1],))
        xprime[idx_no_examples] = np.tile(np.inf, (x.shape[1],))

        return xprime

    def explain_batch(self, x: np.ndarray, y):
        if self.perform_transform:
            x = self.float_transformer.transform(x)
        # an array for the constructed contrastive examples
        X_test = pd.DataFrame(x)
        X_test.columns = self.col_names[:-1]
        xprime = np.empty(shape=x.shape)
        xprime[:, :] = np.inf
        dataset_info: DatasetInfo = self.dataset_info
        rf = self.manager.random_forest.model
        X_train = self.Xtrain

        valids = []
        progress = tqdm(total=x.shape[0], desc="RFOCSE", leave=False)
        for idx, res in zip(X_test.index, batch_extraction(rf, dataset_info, X_test.values, max_distance=100, log_every=-1, max_iterations=20_000_000,
                                                           export_rules_file=None, dataset=X_train.values)):
            if 'explanation' in res:
                cs = res.pop('explanation')
                res['feature_bounds'] = cs.feat_conditions
                res['feature_bounds_compact'] = res.pop('explanation_compact').feat_conditions
                res['feature_bounds_compact_count'] = res.pop('explanation_compact_count').feat_conditions
                res['counterfactual_sample'] = cs.sample_counterfactual(X_test.loc[idx], epsilon=self.offset)
                xprime[idx] = res['counterfactual_sample']
                res['factual_class'] = rf.predict(np.array(X_test.loc[idx]).reshape(1, -1))[0]
                res['counterfactual_distance'] = res['distance']
                res['counterfactual_class'] = rf.predict(np.array(res['counterfactual_sample']).reshape(1, -1))[0]
                res['valid_'] = res['factual_class'] != res['counterfactual_class']
                valids.append(res['valid_'])
            progress.update()
        progress.close()

        # locate non-counterfactual examples and replace them with [np.inf, ... , np.inf]
        idx_no_examples = (xprime == np.inf).any(axis=1)
        xprime[idx_no_examples] = np.tile(0, x.shape[1])
        y_pred = self.manager.predict(xprime)
        idx_failed_explanation = (y_pred == y)
        xprime[idx_failed_explanation] = np.tile(np.inf, (x.shape[1],))
        xprime[idx_no_examples] = np.tile(np.inf, (x.shape[1],))

        return xprime

    def process_pandas_dataset(self, df, y_name, dataset_name='dataset', use_one_hot=False, override_feature_types=None,
                               select_dtypes=None, max_size=None):
        df = df.copy()
        df.dropna(axis=0, inplace=True)
        if override_feature_types is not None:
            for feature, feature_type in override_feature_types.items():
                df[feature] = df[feature].astype(feature_type)

        y = df[y_name]
        X = df.drop(columns=y_name)

        if y.dtype == float:
            raise ValueError("Y class should be an integer or object")
        else:
            unique_values = y.unique()
            values_mapping = dict(zip(unique_values, range(len(unique_values))))
            y = y.apply(lambda x: values_mapping[x])

        if select_dtypes:
            X = X.select_dtypes(select_dtypes)

        X_prep = None

        for feature in X.columns:
            if len(X[feature].unique()) <= 1:
                print("Column {feature} has been removed because it is constant".format(feature=feature))
                X.drop(columns=feature, inplace=True)

        max_size = None
        if max_size is not None and len(X) > max_size:
            X = X.sample(max_size, random_state=0).copy()
            y = y.loc[X.index].copy()

        if self.perform_transform:
            if float in X.dtypes.values:
                float_transformer = StandardScaler()
                float_data = X.select_dtypes(float)
                transformed_float_data = float_transformer.fit_transform(float_data)
                transformed_float_data = pd.DataFrame(data=transformed_float_data, columns=float_data.columns,
                                                      index=float_data.index)
                X.loc[transformed_float_data.index, transformed_float_data.columns] = transformed_float_data
            else:
                float_transformer = None
        else:
            float_transformer = None

        self.float_transformer = float_transformer

        dataset_builder = DatasetInfoBuilder(dataset_name)  # , float_transformer)
        columns = []

        for column, dtype in X.dtypes.to_dict().items():
            current_feature = len(columns)
            column_data = X[column]

            if dtype == int:
                dataset_builder.add_ordinal_variable(current_feature, column_data.min(), column_data.max(),
                                                     name=column)
                columns.append(column)
                new_column_data = column_data
            elif dtype == float:
                dataset_builder.add_numerical_variable(current_feature, column_data.min(), column_data.max(),
                                                       name=column)
                columns.append(column)
                new_column_data = column_data
            elif dtype == object:
                if use_one_hot:
                    column_dummies = pd.get_dummies(column_data, prefix='{}'.format(column))

                    if len(column_dummies.columns) == 2:
                        new_column_data = column_dummies[column_dummies.columns[0]]
                        dataset_builder.add_binary_variable(current_feature, name=column,
                                                            category_names=column_dummies.columns)
                        columns.append(column_dummies.columns[0])
                    else:
                        dataset_builder.add_one_hot_varible(current_feature, len(column_dummies.columns),
                                                            name=column, category_names=column_dummies.columns)
                        new_column_data = column_dummies
                        columns.extend(column_dummies.columns)
                else:
                    unique_values = column_data.unique()
                    unique_values_mapping = dict(zip(sorted(unique_values), range(len(unique_values))))
                    new_column_data = column_data.apply(lambda x: unique_values_mapping[x])
                    columns.append(column)
                    dataset_builder.add_categorical_numerical(current_feature, len(unique_values), name=column,
                                                              category_names=unique_values_mapping)
            else:
                raise ValueError("Type {} in column {} not supported".format(dtype, column))

            if X_prep is None:
                X_prep = pd.DataFrame(new_column_data)
            else:
                X_prep = pd.concat((X_prep, new_column_data), axis=1)

        return dataset_builder.create_dataset_info(), X_prep[columns], y

    def get_dataset_args(self, initial_args, kwargs):
        kwargs = kwargs.copy()

        for key in initial_args:
            if key not in kwargs:
                kwargs[key] = initial_args[key]

        return kwargs


def doRFOCSEExplanationWithMaxTime(maxTime,
                                   problem,
                                   sklearn_rf,
                                   parsed_rf,
                                   dataset_info,
                                   observation,
                                   dataset,
                                   offset,
                                   verbose=False):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=doRFOCSEExplanationWithQueueCatch, args=(q,
                                                                    problem,
                                                                    sklearn_rf,
                                                                    parsed_rf,
                                                                    dataset_info,
                                                                    observation,
                                                                    dataset,
                                                                    offset
                                                                    )
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


def doRFOCSEExplanationWithQueueCatch(queue,
                                      problem,
                                      sklearn_rf,
                                      parsed_rf,
                                      dataset_info,
                                      observation,
                                      dataset,
                                      offset
                                      ):
    try:
        doRFOCSEExplanationWithQueue(queue,
                                     problem,
                                     sklearn_rf,
                                     parsed_rf,
                                     dataset_info,
                                     observation,
                                     dataset,
                                     offset
                                     )
    except:
        print("RFOCSE error")
        queue.put(None)


def doRFOCSEExplanationWithQueue(queue,
                                 problem,
                                 sklearn_rf,
                                 parsed_rf,
                                 dataset_info,
                                 observation,
                                 dataset,
                                 offset
                                 ):
    queue.put(doRFOCSEExplanation(problem,
                                  sklearn_rf,
                                  parsed_rf,
                                  dataset_info,
                                  observation,
                                  dataset,
                                  offset))


def doRFOCSEExplanation(problem,
                        sklearn_rf,
                        parsed_rf,
                        dataset_info,
                        observation,
                        dataset,
                        offset):
    res = extract_single_counterfactual_set(problem, sklearn_rf, parsed_rf, dataset_info, observation, dataset)
    if 'explanation' in res:
        cs = res.pop('explanation')
        return cs.sample_counterfactual(observation, offset)
    else:
        return None
