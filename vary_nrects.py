def vary_nrects(ds_names, nrects=[1_000, 10_000, 50_000, 100_000], iterations=[0, 1, 2, 3, 4]):
    '''
    Experiment to observe the effect of the the number of features on explanation
    '''
    print("Varying number of trees:")
    print("\tds_names:", ds_names)
    print("\tntrees:", ntrees)
    print("\titerations:", iterations)

    experiment_path = "./results/vary-nrects/"
    ntrees = 100
    max_depth = None
    rf_params = {
        "rf_maxdepth": max_depth,
        "rf_ntrees": ntrees,
        "rf_hardvoting": False,  # note OCEAN and FACETIndex use soft and hard requirment
    }
    facet_params = {
        "facet_offset": 0.0001,
        "facet_nrects": -1,
        "facet_sample": "Augment",
        "facet_enumerate": "PointBased",
        "facet_verbose": False,
        "facet_sd": 0.3,
        "facet_search": "BitVector",
        "rbv_initial_radius": 0.01,
        "rbv_radius_growth": "Linear",
        "rbv_num_interval": 4
    }
    params = {
        "RandomForest": rf_params,
        "FACETIndex": facet_params,
    }

    total_runs = len(ds_names) * len(ntrees) * len(iterations)
    progress_bar = tqdm(total=total_runs, desc="Overall Progress", position=0, disable=False)

    # experiment_results = pd.DataFrame(columns=[
    #     "dataset",
    #     "explainer",
    #     "n_trees",
    #     "max_depth",
    #     "iteration",
    #     "n_rects",
    #     "accuracy",
    #     "precision",
    #     "recall",
    #     "f1",
    #     "per_valid",
    #     "avg_dist",
    #     "avg_length",
    #     "prep_time",
    #     "explain_time",
    #     "sample_time",
    #     "n_explain",
    # ])

    for ds in ds_names:
        for nr in nrects:
            for iter in iterations:
                # set the number of rectangles
                params["FACETIndex"]["facet_nrects"] = nr
                # execute the run
                run_result = execute_run(
                    dataset_name=ds,
                    explainer="FACETIndex",
                    params=params,
                    output_path=experiment_path,
                    iteration=iter,
                    test_size=0.2,
                    n_explain=20,
                    random_state=1,
                    preprocessing="Normalize"
                )
                df_item = {
                    "dataset": ds,
                    "explainer": "FACETIndex",
                    "n_trees": ntrees,
                    "max_depth": max_depth,
                    "iteration": iter,
                    "n_rects": nr,
                    **run_result
                }
                experiment_results = pd.DataFrame([df_item])
                # experiment_results = experiment_results.append(df_item, ignore_index=True)
                if not os.path.exists(experiment_path + "vary_ntrees.csv"):
                    experiment_results.to_csv(experiment_path + "vary_ntrees.csv", index=False)
                else:
                    experiment_results.to_csv(experiment_path + "vary_ntrees.csv", index=False, mode="a", header=False,)

                progress_bar.update()
    progress_bar.close()
    print("Finished varying number of trees")
