import argparse

from experiments.auto_figures import (nb_compare_methods, nb_compare_methods_gbc, nb_index_evaluation,
                                      nb_perturbation_robustness, nb_user_workbooks, nb_vary_ntrees)
from experiments.runall_paper import (runall, runall_compare_methods, runall_gradient_boosting_ensemble,
                                      runall_index_scaling, runall_index_vs_linear, runall_ntrees,
                                      runall_personalization, runall_perturbations)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run FACET Experiments')
    # general parameters
    parser.add_argument("--all_results", action="store_true")
    parser.add_argument("--all_iterations", action="store_true")
    parser.add_argument("--all_baselines", action="store_true")
    # specific results
    parser.add_argument("--tab3", action="store_true")
    parser.add_argument("--tab4", action="store_true")
    parser.add_argument("--fig9", action="store_true")
    parser.add_argument("--fig10", action="store_true")
    parser.add_argument("--fig11", action="store_true")
    parser.add_argument("--fig12", action="store_true")
    parser.add_argument("--fig13_14", action="store_true")
    # parse all args
    args = parser.parse_args()

    FAST = not args.all_iterations
    SKIP = not args.all_baselines

    print("-----------------------------------------")
    print("Replicating FACET paper results...")
    if FAST and SKIP:
        print("We've skipped some iterations and left off the slowest baseline methods to save you time!")
        print("For 100% matching results use '--all_iterations' (slow) and '--all_baselines' (VERY slow)")
    if not FAST:
        print("You've diabled 'fast' mode which runs slower experiments for 1 iteration instead of 10. This will take a long time")
    if not SKIP:
        print("You've disabled 'skipslow' mode which skips running the MACE and RFOCSE baseline methods which are INCREDIBLY slow. This will take a VERY long time")

    if args.all_results:
        runall(FAST=FAST, SKIP_SLOW_METHODS=SKIP)
        print("Creating all figures and  tables...")
        nb_compare_methods()
        nb_compare_methods_gbc()
        nb_perturbation_robustness()
        nb_user_workbooks()
        nb_index_evaluation()
        nb_index_evaluation()
        nb_vary_ntrees()
        print("All figures and tables saved!")

    else:
        if args.tab3:
            runall_compare_methods(FAST=FAST, SKIP_SLOW_METHODS=SKIP)
            print("Creating table 3...")
            nb_compare_methods()
            print("Table 3 saved!")
        if args.tab4:
            runall_gradient_boosting_ensemble(FAST=FAST)
            print("Creating table 4...")
            nb_compare_methods_gbc()
            print("Table 4 saved!")
        if args.fig9:
            runall_perturbations(FAST=FAST, SKIP_SLOW_METHODS=SKIP)
            print("Creating figure 9...")
            nb_perturbation_robustness()
            print("Figure 9 saved!")
        if args.fig10:
            runall_personalization(FAST=FAST)
            print("Creating figure 10...")
            nb_user_workbooks()
            print("Figure 10 saved!")
        if args.fig11:
            runall_index_scaling(FAST=FAST)
            print("Creating figure 11...")
            nb_index_evaluation()
            print("Figure 11 saved!")
        if args.fig12:
            runall_index_vs_linear(FAST=FAST)
            print("Creating figure 12...")
            nb_index_evaluation()
            print("Figure 12 saved!")
        if args.fig13_14:
            runall_ntrees(FAST=FAST)
            print("Creating figures 13 and 14...")
            nb_vary_ntrees()
            print("Figures 13 and 14 saved!")
