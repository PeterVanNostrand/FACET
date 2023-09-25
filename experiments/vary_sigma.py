import os
import pandas as pd
from tqdm.auto import tqdm

from .experiments import execute_run, RF_DEFAULT_PARAMS, FACET_DEFAULT_PARAMS, FACET_TUNED_M


def vary_sigma(ds_names, sigmas=[0.01, 0.05, 0.1, 0.2, 0.3], iterations=[0, 1, 2, 3, 4], fmod=None, ntrees=10, max_depth=5):
    '''
    Experiment to observe the effect of the standard deviation of data augmentation on explanation qualtiy
    '''
    print("Varying sigma:")
    print("\tds_names:", ds_names)
    print("\tsigmas:", sigmas)
    print("\titerations:", iterations)

    if fmod is not None:
        csv_path = "./results/vary_sigma_" + fmod + ".csv"
        experiment_path = "./results/vary-sigma-" + fmod + "/"
    else:
        csv_path = "./results/vary_sigma.csv"
        experiment_path = "./results/vary-sigma/"

    explainer = "FACETIndex"
    params = {
        "RandomForest": RF_DEFAULT_PARAMS,
        "FACETIndex": FACET_DEFAULT_PARAMS,
    }
    params["FACETIndex"]["facet_sd"] = -1
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth

    total_runs = len(ds_names) * len(sigmas) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for sig in sigmas:
            for ds in ds_names:
                # set the number of trees
                params["FACETIndex"]["facet_sd"] = sig
                params["FACETIndex"]["rbv_num_interval"] = FACET_TUNED_M[ds]
                run_result = execute_run(
                    dataset_name=ds,
                    explainer=explainer,
                    params=params,
                    output_path=experiment_path,
                    iteration=iter,
                    test_size=0.2,
                    n_explain=20,
                    random_state=iter,
                    preprocessing="Normalize",
                    run_ext="sig{:.4f}_".format(sig)
                )
                df_item = {
                    "dataset": ds,
                    "explainer": explainer,
                    "n_trees": params["RandomForest"]["rf_ntrees"],
                    "max_depth": params["RandomForest"]["rf_maxdepth"],
                    "facet_sd": sig,
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
    print("Finished varying sigma")
