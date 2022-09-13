import os
import pandas as pd
from tqdm.auto import tqdm

from experiments import execute_run


def vary_sigma(ds_names, sigmas=[0.01, 0.05, 0.1, 0.2, 0.3], iterations=[0, 1, 2, 3, 4]):
    '''
    Experiment to observe the effect of the standard deviation of data augmentation on explanation qualtiy
    '''
    print("Varying sigma:")
    print("\tds_names:", ds_names)
    print("\tsigmas:", sigmas)
    print("\titerations:", iterations)

    csv_path = "./results/vary_sigma_500t.csv"
    experiment_path = "./results/vary-sigma-500t/"
    explainer = "FACETIndex"
    ntrees = 500
    max_depth = None
    rf_params = {
        "rf_maxdepth": max_depth,
        "rf_ntrees": ntrees,
        "rf_hardvoting": True
    }
    facet_params = {
        "facet_offset": 0.001,
        "facet_nrects": 20_000,
        "facet_sample": "Augment",
        "facet_enumerate": "PointBased",
        "facet_verbose": False,
        "facet_sd": -1,
        "facet_search": "BitVector",
        "rbv_initial_radius": 0.01,
        "rbv_radius_growth": "Linear",
        "rbv_num_interval": 4
    }
    params = {
        "RandomForest": rf_params,
        "FACETIndex": facet_params,
    }

    total_runs = len(ds_names) * len(sigmas) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for sig in sigmas:
            for ds in ds_names:
                # set the number of trees
                params["FACETIndex"]["facet_sd"] = sig
                run_result = execute_run(
                    dataset_name=ds,
                    explainer=explainer,
                    params=params,
                    output_path=experiment_path,
                    iteration=iter,
                    test_size=0.2,
                    n_explain=20,
                    random_state=1,
                    preprocessing="Normalize",
                    run_ext="sig{:.4f}_".format(sig)
                )
                df_item = {
                    "dataset": ds,
                    "explainer": explainer,
                    "n_trees": ntrees,
                    "max_depth": max_depth,
                    "facet_sd": sig,
                    "iteration": iter,
                    "max_depth": max_depth,
                    **run_result
                }
                experiment_results = pd.DataFrame([df_item])
                if not os.path.exists(csv_path):
                    experiment_results.to_csv(csv_path, index=False)
                else:
                    experiment_results.to_csv(csv_path, index=False, mode="a", header=False,)

                progress_bar.update()
    progress_bar.close()
    print("Finished varying sigma")
