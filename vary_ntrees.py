import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm.auto import tqdm

from experiments import execute_run

tuned_facet_sd = {
    "cancer": 0.1,
    "glass": 0.1,
    "magic": 0.005,
    "spambase": 0.01,
    "vertebral": 0.05
}


def vary_ntrees(ds_names, explainers=["FACETIndex", "OCEAN", "RFOCSE", "AFT", "MACE"], ntrees=[5, 10, 15],
                iterations=[0, 1, 2, 3, 4]):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    print("Varying number of trees:")
    print("\tds_names:", ds_names)
    print("\texplainers:", explainers)
    print("\tntrees:", ntrees)
    print("\titerations:", iterations)

    csv_path = "./results/vary_ntrees_sigtuned.csv"
    experiment_path = "./results/vary-ntrees-sigtuned/"
    max_depth = 5
    rf_params = {
        "rf_maxdepth": max_depth,
        "rf_ntrees": -1,
        "rf_hardvoting": False,  # note OCEAN and FACETIndex use soft and hard requirment
    }
    facet_params = {
        "facet_offset": 0.001,
        "facet_nrects": 20000,
        "facet_sample": "Augment",
        "facet_enumerate": "PointBased",
        "facet_verbose": False,
        "facet_sd": 0.3,
        "facet_search": "BitVector",
        "rbv_initial_radius": 0.01,
        "rbv_radius_growth": "Linear",
        "rbv_num_interval": 4
    }
    rfocse_params = {
        "rfoce_transform": False,
        "rfoce_offset": 0.0001
    }
    aft_params = {
        "aft_offset": 0.0001
    }
    mace_params = {
        "mace_maxtime": 300,
        "mace_epsilon": 0.001
    }
    ocean_params = {
        "ocean_norm": 2
    }
    params = {
        "RandomForest": rf_params,
        "FACETIndex": facet_params,
        "MACE": mace_params,
        "RFOCSE": rfocse_params,
        "AFT": aft_params,
        "OCEAN": ocean_params,
    }

    total_runs = len(ds_names) * len(explainers) * len(ntrees) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for ntree in ntrees:
            for expl in explainers:
                for ds in ds_names:
                    # set the number of trees
                    params["RandomForest"]["rf_ntrees"] = ntree
                    params["FACETIndex"]["facet_sd"] = tuned_facet_sd[ds]
                    # FACET uses hardvoting
                    if expl == "FACETIndex":
                        params["RandomForest"]["rf_hardvoting"] = True
                    else:
                        params["RandomForest"]["rf_hardvoting"] = False
                    run_result = execute_run(
                        dataset_name=ds,
                        explainer=expl,
                        params=params,
                        output_path=experiment_path,
                        iteration=iter,
                        test_size=0.2,
                        n_explain=20,
                        random_state=1,
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
