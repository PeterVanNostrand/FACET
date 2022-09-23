import os
import pandas as pd
from tqdm.auto import tqdm

from experiments import execute_run
from experiments import TUNED_FACET_SD, FACET_DEFAULT_PARAMS, RF_DEFAULT_PARAMS


def vary_nrects(ds_names, nrects=[5, 10, 15], iterations=[0, 1, 2, 3, 4], fmod=None):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    print("Varying number of hyperrectangles:")
    print("\tds_names:", ds_names)
    print("\tnrects:", nrects)
    print("\titerations:", iterations)

    if fmod is not None:
        csv_path = "./results/vary_nrects_" + fmod + ".csv"
        experiment_path = "./results/vary-nrects-" + fmod + "/"
    else:
        csv_path = "./results/vary_nrects.csv"
        experiment_path = "./results/vary-nrects/"

    explainer = "FACETIndex"
    params = {
        "RandomForest": RF_DEFAULT_PARAMS,
        "FACETIndex": FACET_DEFAULT_PARAMS,
    }
    params["FACETIndex"]["facet_nrects"] = -1

    total_runs = len(ds_names) * len(nrects) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for nr in nrects:
            for ds in ds_names:
                # set the number of trees
                params["FACETIndex"]["facet_nrects"] = nr
                params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds]
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
                    run_ext="r{:03d}_".format(nr)
                )
                df_item = {
                    "dataset": ds,
                    "explainer": explainer,
                    "n_trees": params["RandomForest"]["rf_ntrees"],
                    "max_depth": params["RandomForest"]["rf_maxdepth"],
                    "n_rects": nr,
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
    print("Finished varying number of rectangle")
