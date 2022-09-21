from unittest.mock import DEFAULT
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm.auto import tqdm

from experiments import execute_run
from experiments import TUNED_FACET_SD, DEFAULT_PARAMS


def compare_methods(ds_names, explainers=["FACETIndex", "OCEAN", "RFOCSE", "AFT", "MACE"], iterations=[0, 1, 2, 3, 4], fmod=None):
    '''
    Experiment to compare the performance of different explainers on the same ensemble
    '''
    print("Comparing methods:")
    print("\tds_names:", ds_names)
    print("\texplainers:", explainers)
    print("\titerations:", iterations)

    if fmod is not None:
        csv_path = "./results/compare_methods_" + fmod + ".csv"
        experiment_path = "./results/compare-methods-" + fmod + "/"
    else:
        csv_path = "./results/compare_methods.csv"
        experiment_path = "./results/compare-methods/"

    ntrees = 10
    max_depth = 5
    params = DEFAULT_PARAMS
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth

    total_runs = len(ds_names) * len(explainers) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for expl in explainers:
            for ds in ds_names:
                # set the number of trees
                params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds]
                run_result = execute_run(
                    dataset_name=ds,
                    explainer=expl,
                    params=params,
                    output_path=experiment_path,
                    iteration=iter,
                    test_size=0.2,
                    n_explain=20,
                    random_state=iter,
                    preprocessing="Normalize",
                    run_ext=""
                )
                df_item = {
                    "dataset": ds,
                    "explainer": expl,
                    "n_trees": ntrees,
                    "max_depth": max_depth,
                    "iteration": iter,
                    **run_result
                }
                experiment_results = pd.DataFrame([df_item])
                if not os.path.exists(csv_path):
                    experiment_results.to_csv(csv_path, index=False)
                else:
                    experiment_results.to_csv(csv_path, index=False, mode="a", header=False,)

                progress_bar.update()
    progress_bar.close()
    print("Finished comparing methods!")