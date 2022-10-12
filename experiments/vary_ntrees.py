import pandas as pd
import os
from tqdm.auto import tqdm

from .experiments import execute_run
from .experiments import TUNED_FACET_SD, FACET_DEFAULT_PARAMS, OCEAN_DEFAULT_PARAMS, RF_DEFAULT_PARAMS, MACE_DEFAULT_PARAMS, FACET_TUNED_M


def vary_ntrees(ds_names, explainers=["FACETIndex", "OCEAN"], ntrees=[5, 10, 15],
                iterations=[0, 1, 2, 3, 4], fmod=None, max_depth=5):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    print("Varying number of trees:")
    print("\tds_names:", ds_names)
    print("\texplainers:", explainers)
    print("\tntrees:", ntrees)
    print("\titerations:", iterations)

    if fmod is not None:
        csv_path = "./results/vary_ntrees_" + fmod + ".csv"
        experiment_path = "./results/vary-ntrees-" + fmod + "/"
    else:
        csv_path = "./results/vary_ntrees.csv"
        experiment_path = "./results/vary-ntrees/"

    params = {
        "RandomForest": RF_DEFAULT_PARAMS,
        "FACETIndex": FACET_DEFAULT_PARAMS,
        "OCEAN": OCEAN_DEFAULT_PARAMS,
        "MACE": MACE_DEFAULT_PARAMS
    }
    params["RandomForest"]["rf_ntrees"] = -1
    params["RandomForest"]["rf_maxdepth"] = max_depth

    total_runs = len(ds_names) * len(explainers) * len(ntrees) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for ntree in ntrees:
            for expl in explainers:
                for ds in ds_names:
                    # set the number of trees
                    params["RandomForest"]["rf_ntrees"] = ntree
                    params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds]
                    params["FACETIndex"]["rbv_num_interval"] = FACET_TUNED_M[ds]

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
                        run_ext="t{:03d}_".format(ntree)
                    )
                    df_item = {
                        "dataset": ds,
                        "explainer": expl,
                        "n_trees": ntree,
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
    print("Finished varying number of trees")
