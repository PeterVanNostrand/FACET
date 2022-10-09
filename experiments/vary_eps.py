import os
import pandas as pd
from tqdm.auto import tqdm

from .experiments import execute_run, RF_DEFAULT_PARAMS, MACE_DEFAULT_PARAMS


def vary_eps(ds_names, epsilons=[1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10], iterations=[0], fmod=None, ntrees=10, max_depth=5):
    '''
    Experiment to observe the effect of the the step size on MACE's runtime
    '''
    print("Varying sigma:")
    print("\tds_names:", ds_names)
    print("\tepsilons:", epsilons)
    print("\titerations:", iterations)

    if fmod is not None:
        csv_path = "./results/vary_eps_" + fmod + ".csv"
        experiment_path = "./results/vary-eps-" + fmod + "/"
    else:
        csv_path = "./results/vary_eps.csv"
        experiment_path = "./results/vary-eps/"

    explainer = "MACE"
    params = {
        "RandomForest": RF_DEFAULT_PARAMS,
        "MACE": MACE_DEFAULT_PARAMS,
    }
    params["MACE"]["mace_epsilon"] = -1
    params["RandomForest"]["rf_ntrees"] = ntrees
    params["RandomForest"]["rf_maxdepth"] = max_depth

    total_runs = len(ds_names) * len(epsilons) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    for iter in iterations:
        for eps in epsilons:
            for ds in ds_names:
                # set the number of trees
                params["MACE"]["mace_epsilon"] = eps
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
                    run_ext="eps{:.0e}_".format(eps)
                )
                df_item = {
                    "dataset": ds,
                    "explainer": explainer,
                    "n_trees": params["RandomForest"]["rf_ntrees"],
                    "max_depth": params["RandomForest"]["rf_maxdepth"],
                    "mace_epsilon": eps,
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
    print("Finished varying epsilon")
