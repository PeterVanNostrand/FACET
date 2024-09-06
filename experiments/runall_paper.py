

from experiments.compare_methods import compare_methods
from experiments.perturbations import perturb_explanations
from experiments.vary_k import vary_k
from experiments.vary_m import vary_m
from experiments.vary_nconstraints import vary_nconstraints
from experiments.vary_nrects import vary_nrects
from experiments.vary_robustness import vary_min_robustness

paper_datasets = ["adult", "cancer", "compas", "credit", "glass", "magic", "spambase", "vertebral"]
paper_explainers = ["FACET", "OCEAN", "RFOCSE", "AFT", "MACE"]


def runall_compare_methods(FAST, SKIP_SLOW_METHODS):
    '''
    Run the compare methods experiments shown in Table 3
    '''
    print("-----------------------------------------")
    print("Generating result for table 3 and table 3 appendix...")

    ntrees = 10
    max_depth = 5
    fmod = "tab3"
    time_limit = 300

    # ########## MAIN PAPER DATASETS ###########
    # all methods except RFOCSE
    compare_methods(
        ds_names=["adult", "cancer", "credit", "magic", "spambase"],
        explainers=["FACET", "OCEAN", "AFT"],
        iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        max_time=time_limit,
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
    )
    if not SKIP_SLOW_METHODS:
        compare_methods(
            ds_names=["adult", "cancer", "credit", "magic", "spambase"],
            explainers=["MACE"],
            iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            max_time=time_limit,
            fmod=fmod,
            ntrees=ntrees,
            max_depth=max_depth,
        )

    if not SKIP_SLOW_METHODS:
        # RFOCSE for tractably fast datasets
        compare_methods(
            ds_names=["adult", "credit", "magic"],
            explainers=["RFOCSE"],
            iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            max_time=time_limit,
            fmod=fmod,
            ntrees=ntrees,
            max_depth=max_depth,
        )

        # run RFOCSE for slow datasets w/uncapped time
        compare_methods(
            ds_names=["cancer", "spambase"],
            explainers=["RFOCSE"],
            iterations=[0],
            max_time=None,
            fmod=fmod,
            ntrees=ntrees,
            max_depth=max_depth,
        )

    # ########## APPENDIX DATASETS ###########
    # all methods
    compare_methods(
        ds_names=["compas", "glass", "vertebral"],
        explainers=["FACET", "OCEAN", "AFT"],
        iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        max_time=time_limit,
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
    )
    if not SKIP_SLOW_METHODS:
        compare_methods(
            ds_names=["compas", "glass", "vertebral"],
            explainers=["MACE", "RFOCSE"],
            iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            max_time=time_limit,
            fmod=fmod,
            ntrees=ntrees,
            max_depth=max_depth,
        )

    print("Results for table 3 and table 3 appendix done!")


def runall_gradient_boosting_ensemble(FAST):
    print("-----------------------------------------")
    print("Generating result for table 4 and table 4 appendix...")

    fmod = "tab4"
    ntrees = 100
    max_depth = 3

    # run all main paper datasets, and apdx datasets together
    # FACET explaining random forest model
    compare_methods(
        ds_names=["adult", "cancer", "credit", "magic", "spambase", "compas", "glass", "vertebral"],
        explainers=["FACET"],
        iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        model_type="RandomForest"
    )

    # FACET explaining gradient boosting classifier with full interesection strategy
    compare_methods(
        ds_names=["adult", "cancer", "credit", "magic", "spambase", "compas", "glass", "vertebral"],
        explainers=["FACET"],
        iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        model_type="GradientBoostingClassifier",
        gbc_intersection="CompleteEnsemble"
    )

    # FACET explaining gradient boosting classifier with relaxed intersection strategy
    compare_methods(
        ds_names=["adult", "cancer", "credit", "magic", "spambase", "compas", "glass", "vertebral"],
        explainers=["FACET"],
        iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        model_type="GradientBoostingClassifier",
        gbc_intersection="MinimalWorstGuess"
    )

    print("Results for table 4 and table 4 appendix done!")


def runall_perturbations(FAST, SKIP_SLOW_METHODS):
    print("-----------------------------------------")
    print("Generating result for figure 9 and figure 9 appendix...")

    fmod = "fig9"
    ntrees = 10
    max_depth = 5
    pert_sizes = [0.0001, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045,
                  0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]

    # all methods on adult and magic
    perturb_explanations(
        ds_names=["adult", "magic", "compas", "glass", "vertebral"],
        explainers=["FACET", "OCEAN", "AFT"],
        pert_sizes=pert_sizes,
        iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        max_time=None,
    )
    if not SKIP_SLOW_METHODS:
        perturb_explanations(
            ds_names=["adult", "magic", "compas", "glass", "vertebral"],
            explainers=["MACE", "RFOCSE"],
            pert_sizes=pert_sizes,
            iterations=[0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            fmod=fmod,
            ntrees=ntrees,
            max_depth=max_depth,
            max_time=None,
        )

    # skip MACE and RFOCSE for credit as they fail often
    perturb_explanations(
        ds_names=["credit"],
        explainers=["FACET", "OCEAN", "AFT"],
        pert_sizes=pert_sizes,
        iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        max_time=None,
    )

    # use different iterations for cancer to avoid OCEAN bug on some seeds
    perturb_explanations(
        ds_names=["cancer"],
        explainers=["FACET", "OCEAN", "AFT"],
        pert_sizes=pert_sizes,
        iterations=[0, 2, 4, 6, 7, 18, 21, 24, 30, 35],
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        max_time=None,
    )
    if not SKIP_SLOW_METHODS:
        perturb_explanations(
            ds_names=["cancer"],
            explainers=["MACE", "RFOCSE"],
            pert_sizes=pert_sizes,
            iterations=[0] if FAST else [0, 2, 4, 6, 7, 18, 21, 24, 30, 35],
            fmod=fmod,
            ntrees=ntrees,
            max_depth=max_depth,
            max_time=None,
        )

    # use different iterations for spambase to avoid OCEAN bug on some seeds
    perturb_explanations(
        ds_names=["spambase"],
        explainers=["FACET", "OCEAN", "AFT"],
        pert_sizes=pert_sizes,
        iterations=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        max_time=None,
    )
    if not SKIP_SLOW_METHODS:
        perturb_explanations(
            ds_names=["spambase"],
            explainers=["MACE", "RFOCSE"],
            pert_sizes=pert_sizes,
            iterations=[0] if FAST else [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            fmod=fmod,
            ntrees=ntrees,
            max_depth=max_depth,
            max_time=None,
        )

    print("Results for figure 9 and figure 9 appendix done!")


def runall_personalization(FAST):
    print("-----------------------------------------")
    print("Generating result for figure 10 and figure 10 appendix...")

    all_ds = ["adult", "cancer", "credit", "magic", "spambase", "compas", "glass", "vertebral",]
    fmod = "fig10"
    ntrees = 100
    max_depth = None
    iterations = [0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Subfigure (a, b ,c) - test the minimum robustness parameter for subfigures
    min_roubust_vals = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
                        0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]
    vary_min_robustness(
        ds_names=all_ds,
        min_robust=min_roubust_vals,
        iterations=iterations,
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth
    )

    # ###### Subfigure (d) - varying number of constraints
    nconstraints_vals = {
        "adult": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "cancer": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
        "credit": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "magic": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "spambase": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "compas": [0, 2, 4, 6, 8, 10],
        "vertebral": [0, 2, 4, 6, 8, 10, 12],
        "glass": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    }
    for ds in all_ds:
        vary_nconstraints(
            ds_names=[ds],
            nconstraints=nconstraints_vals[ds],
            iterations=iterations,
            fmod=fmod, ntrees=ntrees,
            max_depth=max_depth
        )

    # Subfigure (e) - varying, k the number of explanations
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    vary_k(
        ds_names=all_ds,
        ks=k_values,
        iterations=iterations,
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth
    )

    print("Results for figure 10 and figure 10 appendix done!")
    pass


def runall_index_scaling(FAST):
    print("-----------------------------------------")
    print("Generating result for figure 11 and figure 11 appendix...")

    ntrees = 100
    max_depth = None
    fmod = "fig11"

    # ##### Subfigures (a, b, c, d) - varying the number of FACET's hyperrectangles
    index_splits_per_axis = 4
    iterations = [0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    all_ds = ["adult", "cancer", "credit", "magic", "spambase", "compas", "glass", "vertebral"]
    nrects_vals = [1_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]

    vary_nrects(
        ds_names=all_ds,
        nrects=nrects_vals,
        iterations=iterations,
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        m=index_splits_per_axis,
        facet_search="BitVector"
    )

    # ##### Subfigure (e) - varying the number of split values used in FACET's index
    nsplit_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    vary_m(
        ds_names=all_ds,
        ms=nsplit_vals,
        iterations=iterations,
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth
    )

    print("Results for figure 11 and figure 11 appendix done!")


def runall_index_vs_linear(FAST):
    print("-----------------------------------------")
    print("Generating result for figure 12 and figure 12 appendix...")

    ntrees = 100
    max_depth = 5
    fmod = "fig12"

    iterations = [0] if FAST else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    all_ds = ["adult", "cancer", "credit", "magic", "spambase", "compas", "glass", "vertebral"]
    nrects_vals = [1_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000]

    # vary nrects with a straight linear search
    vary_nrects(
        ds_names=all_ds,
        nrects=nrects_vals,
        iterations=iterations,
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        m=None,
        facet_search="Linear"
    )

    # vary nrects with a FACET's BitVector index
    vary_nrects(
        ds_names=all_ds,
        nrects=nrects_vals,
        iterations=iterations,
        fmod=fmod,
        ntrees=ntrees,
        max_depth=max_depth,
        m=None,
        facet_search="BitVector"
    )

    print("Results for figure 12 and figure 12 appendix done!")


def runall(FAST=False, SKIP_SLOW_METHODS=False):
    print("Executing all paper experiments, this may take a while...")
    runall_compare_methods(FAST=FAST, SKIP_SLOW_METHODS=SKIP_SLOW_METHODS)
    runall_gradient_boosting_ensemble(FAST=FAST)
    runall_perturbations(FAST=FAST, SKIP_SLOW_METHODS=SKIP_SLOW_METHODS)
    runall_personalization(FAST=FAST)
