import os
import pandas as pd
from tqdm.auto import tqdm

from experiments import execute_run
from experiments import TUNED_FACET_SD


def vary_enum(ds_names, iterations=[0, 1, 2, 3, 4], fmod=None):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    hard_votings = [False, True]
    enumerations = ["Size", "Axes", "Probability"]

    print("Testing hyperrectangle enumeration methods:")
    print("\tds_names:", ds_names)
    print("\titerations:", iterations)
    print("\tvotings:", hard_votings)
    print("\tenumerations:", enumerations)

    if fmod is not None:
        csv_path = "./results/vary_enum_" + fmod + ".csv"
        experiment_path = "./results/vary-enum-" + fmod + "/"
    else:
        csv_path = "./results/vary_enum.csv"
        experiment_path = "./results/vary-enum/"

    explainer = "FACETIndex"
    ntrees = 100
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
        "rbv_num_interval": 4,
    }
    params = {
        "RandomForest": rf_params,
        "FACETIndex": facet_params,
    }

    total_runs = len(ds_names) * len(hard_votings) * len(enumerations) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for vote in hard_votings:
            for enum in enumerations:
                for ds in ds_names:
                    params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds]
                    params["RandomForest"]["rf_hardvoting"] = vote
                    params["FACETIndex"]["facet_intersect_order"] = enum
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
                        run_ext="h{}_e{}_".format(vote, enum.lower())
                    )
                    df_item = {
                        "dataset": ds,
                        "explainer": explainer,
                        "n_trees": ntrees,
                        "max_depth": max_depth,
                        "hard_voting": vote,
                        "enumeration": enum,
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
    print("Finished testing enumeration methods")
