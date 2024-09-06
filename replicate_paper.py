import argparse
import json
import os
import re

from experiments.runall_paper import (runall, runall_compare_methods, runall_gradient_boosting_ensemble,
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
    # parse all args
    args = parser.parse_args()

    FAST = not args.all_iterations
    SKIP = not args.all_baselines

    print("-----------------------------------------")
    print("Replicating FACET paper results...")
    if FAST and SKIP:
        print("We've skipped some iterations and left off the slowest baseline methods to save you time!")
        print("For 100% complete results see '--all_iterations' and 'all_baselines'")
    if not FAST:
        print("You've diabled 'fast' mode which runs slower experiments for 1 iteration instead of 10. This will take a long time")
    if not SKIP:
        print("You've disabled 'skipslow' mode which skips running the MACE and RFOCSE baseline methods which are INCREDIBLY slow. This will take a VERY long time")

    if args.all_results:
        runall(FAST=FAST, SKIP_SLOW_METHODS=SKIP)
    else:
        if args.tab3:
            runall_compare_methods(FAST=FAST, SKIP_SLOW_METHODS=SKIP)
        if args.tab4:
            runall_gradient_boosting_ensemble(FAST=FAST)
        if args.fig9:
            runall_perturbations(FAST=FAST, SKIP_SLOW_METHODS=SKIP)
        if args.fig10:
            runall_personalization(FAST=FAST)
