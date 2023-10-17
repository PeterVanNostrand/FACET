import argparse
import json
import os
import re

from experiments.compare_methods import compare_methods
from experiments.experiments import (DEFAULT_PARAMS, FACET_TUNED_M,
                                     TUNED_FACET_SD, execute_run)
from experiments.vary_enum import vary_enum
from experiments.vary_eps import vary_eps
from experiments.vary_k import vary_k
from experiments.vary_m import vary_m
from experiments.vary_nconstraints import vary_nconstraints
from experiments.vary_nrects import vary_nrects
from experiments.vary_ntrees import vary_ntrees
from experiments.vary_rinit import vary_rinit
from experiments.vary_rstep import vary_rstep
from experiments.vary_sigma import vary_sigma
from experiments.perturbations import perturb_explanations
from experiments.widths import compute_widths
from experiments.vary_robustness import vary_robustness


# TODO: Currently ignoring feature actionability
# TODO: 3. Get MACE running for this data
# TODO: 4. Implement feature typing and actionability to FACET
# TODO: 5. Get RFOCSE working
# TODO: 6. Get AFT working

def check_create_directory(dir_path="./results/"):
    '''
    Checks the the directory at `dir_path` exists, if it does not it creates all directories in the path
    '''
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # find the next availible run_id in the specified results directory
    max_run_id = 0
    dir_names = os.listdir(dir_path)
    for name in dir_names:
        x = re.match("run-(\d{3})", name)  # noqa: W605 (ignore linting from flake)
        if x is not None:
            found_run_id = int(x.group(1))
            if found_run_id > max_run_id:
                max_run_id = found_run_id
    run_id = max_run_id + 1

    # return the run_id and a path to that folder
    run_dir = "run-{:03d}/".format(run_id)
    run_path = os.path.join(os.path.abspath(dir_path), run_dir)
    os.makedirs(run_path)
    return run_id, run_path


def simple_run(ds_name="vertebral", explainer="FACETIndex", random_state=0, ntrees=10,
               max_depth=5, model_type="RandomForest"):
    # Euclidean, FeaturesChanged
    run_id, run_path = check_create_directory("./results/simple-run/")

    params = DEFAULT_PARAMS
    if model_type == "RandomForest":
        params["RandomForest"]["rf_ntrees"] = ntrees
        params["RandomForest"]["rf_maxdepth"] = max_depth
    if model_type == "GradientBoostingClassifier":
        params["GradientBoostingClassifier"]["gbc_ntrees"] = ntrees
        params["GradientBoostingClassifier"]["gbc_maxdepth"] = max_depth

    params["FACETIndex"]["facet_sd"] = TUNED_FACET_SD[ds_name] if ds_name in TUNED_FACET_SD else 0.01
    params["FACETIndex"]["rbv_num_interval"] = FACET_TUNED_M[ds_name] if ds_name in FACET_TUNED_M else 16

    preprocessing = "Normalize"
    n_explain = 20

    print("Run ID: {}".format(run_id))
    print("explainer: " + explainer)
    print("dataset: " + ds_name)
    print("config:")
    print(json.dumps(params, indent=4))

    results = execute_run(
        dataset_name=ds_name,
        explainer=explainer,
        params=params,
        output_path=run_path,
        iteration=random_state,
        test_size=0.2,
        n_explain=n_explain,
        random_state=random_state,
        preprocessing=preprocessing,
        model_type=model_type,
    )
    print("results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":

    all_ds = ["cancer", "glass", "magic", "spambase", "vertebral", "adult", "credit", "compas"]
    all_explaiers = ["FACETIndex", "OCEAN", "RFOCSE", "AFT", "MACE"]

    parser = argparse.ArgumentParser(description='Run FACET Experiments')
    parser.add_argument("--expr", choices=["simple", "ntrees", "nrects",
                        "eps", "sigma", "enum", "compare", "k", "rinit", "rstep",
                                           "m", "nconstraints", "perturb", "widths", "robust"], default="simple")
    parser.add_argument("--ds", type=str, nargs="+", default=["vertebral"])
    parser.add_argument("--method", type=str, nargs="+", choices=all_explaiers, default=["FACETIndex"])
    parser.add_argument("--values", type=float, nargs="+", default=None)
    parser.add_argument("--ntrees", type=int, default=10)
    parser.add_argument("--maxdepth", type=int, default=5)
    parser.add_argument("--it", type=int, nargs="+", default=[0])
    parser.add_argument("--fmod", type=str, default=None)
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "gbc"])
    args = parser.parse_args()

    # Set maxdepth to -1 to allow trees to grow uncapped
    if args.maxdepth == -1:
        args.maxdepth = None

    # parse the long version of the model string
    MODEL_TYPES = {
        "rf": "RandomForest",
        "gbc": "GradientBoostingClassifier",
    }
    args.model = MODEL_TYPES[args.model]

    print(args)

    # Do a single quick run with one explaienr and one dataset
    if args.expr == "simple":
        simple_run(ds_name=args.ds[0], explainer=args.method[0],
                   random_state=args.it[0], ntrees=args.ntrees, max_depth=args.maxdepth, model_type=args.model)
    # Vary the number of trees and compare explaienrs
    elif args.expr == "ntrees":
        if args.values is not None:
            ntrees = [int(_) for _ in args.values]
            vary_ntrees(ds_names=args.ds, explainers=args.method, ntrees=ntrees,
                        iterations=args.it, fmod=args.fmod, max_depth=args.maxdepth)
        else:
            vary_ntrees(ds_names=args.ds, explainers=args.method,
                        iterations=args.it, fmod=args.fmod, max_depth=args.maxdepth)

    # Vary the number of hyperrectangles for FACETIndex
    elif args.expr == "nrects":
        if args.values is not None:
            nrects = [int(_) for _ in args.values]
            vary_nrects(ds_names=args.ds, nrects=nrects, iterations=args.it,
                        fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)
        else:
            vary_nrects(ds_names=args.ds, iterations=args.it, fmod=args.fmod,
                        ntrees=args.ntrees, max_depth=args.maxdepth)

    # Vary the epsilon value for MACE
    elif args.expr == "eps":
        if args.values is not None:
            vary_eps(ds_names=args.ds, epsilons=args.values, iterations=args.it,
                     fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)
        else:
            vary_eps(ds_names=args.ds, iterations=args.it, fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    # Vary the standard deviation of HR enumeration for FACETIndex
    elif args.expr == "sigma":
        if args.values is not None:
            vary_sigma(ds_names=args.ds, sigmas=args.values, iterations=args.it,
                       fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)
        else:
            vary_sigma(ds_names=args.ds, iterations=args.it, fmod=args.fmod,
                       ntrees=args.ntrees, max_depth=args.maxdepth)

    elif args.expr == "enum":
        vary_enum(ds_names=args.ds, iterations=args.it, fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    elif args.expr == "compare":
        compare_methods(ds_names=args.ds, explainers=args.method, iterations=args.it,
                        fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    elif args.expr == "k":
        ks = [int(_) for _ in args.values]
        vary_k(ds_names=args.ds, ks=ks, iterations=args.it,
               fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    elif args.expr == "rinit":
        vary_rinit(ds_names=args.ds, rs=args.values, iterations=args.it,
                   fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    elif args.expr == "rstep":
        vary_rstep(ds_names=args.ds, rs=args.values, iterations=args.it,
                   fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    # vary the number of example the user request
    elif args.expr == "m":
        ms = [int(_) for _ in args.values]
        vary_m(ds_names=args.ds, ms=ms, iterations=args.it,
               fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    # vary the number of constraints the user applies. restrict nconstraints <= 2*nfeatures
    elif args.expr == "nconstraints":
        nconstraints = [int(_) for _ in args.values]
        vary_nconstraints(ds_names=args.ds, nconstraints=nconstraints, iterations=args.it,
                          fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    elif args.expr == "perturb":
        perturb_explanations(ds_names=args.ds, explainers=args.method, pert_sizes=args.values,
                             iterations=args.it, fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)

    elif args.expr == "widths":
        compute_widths(ds_names=args.ds, iteration=args.it[0], ntrees=args.ntrees, max_depth=args.maxdepth)

    elif args.expr == "robust":
        if args.values is not None:
            vary_robustness(ds_names=args.ds, min_robust=args.values, iterations=args.it,
                            fmod=args.fmod, ntrees=args.ntrees, max_depth=args.maxdepth)
        else:
            print("using default values")
            vary_robustness(ds_names=args.ds, iterations=args.it, fmod=args.fmod,
                            ntrees=args.ntrees, max_depth=args.maxdepth)
