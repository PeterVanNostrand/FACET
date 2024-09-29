# Reproducibility Instructions

Use the following instructions to reproduce the results shown in our paper *[FACET: Robust Counterfactual Explanation Analytics](https://petervannostrand.github.io/publication/FACET-Robust-CFs)*

## Reproducibility Checklist

This overview follows the layout of the SIGMOD'24 [ARI quick guide](https://docs.google.com/document/d/1_pheZ2p9Nc8qhtcOpNINm7AxFpPpkpC1n60jJdyr-uk/export?format=pdf&attachment=false)

1. Author's source code
   1. Available in this repository [https://github.com/PeterVanNostrand/FACET](https://github.com/PeterVanNostrand/FACET)
2. Description of software/library dependencies
   1. FACET is implemented in python3, [requirements.txt](/requirements.txt) contains a full list of required packages
   2. This is formatted for use with the [Anaconda](https://www.anaconda.com/download/success) virtual environment and package manager
3. Description of hardware needed
   1. A modern quad-core CPU and 16GB of RAM are sufficient for all experiments
   2. See our [runtime notes](#runtime-notes) on accelerating execution if you have more resources available
4. Detailed fully automated scripts that collect data and plot all figures
   1. The code in `replicate_paper.py` automatically runs all paper experiments and generates matching figures/tables
   2. See the instructions below for usage of `replicate_paper.py`
   3. We support running experiments outside of those in the paper via `main.py` as documented in [our README](/README.md)
5. Documentation on how to compile, deploy, run the code, and use the scripts
   1. See the instructions below for documentation on usage of `replicate_paper.py`
6. Link to a single master script
   1. See the instructions below for usage of `replicate_paper.py`
7. An estimation of the deployment time and effort
   1. See the time estimates in table below

## Directions to Reproduce Experiments

### Step 1: Install Dependencies

- Follow the steps in [installation.md](/instructions/INSTALLATION.md) to install the required dependencies, then return here
- For reproducing the FACET paper you can skip the optional demo UI installation steps as marked

### Step 2: Run Experiments

For your convenience we provide a two options for reproducing the experiments in our paper

- **Option 1:** Run every experiment sequentially with `python replicate_paper.py --all_results` (approx 12.5 hours total)
- **Option 2** Run the experiments for specific table(s) or figure(s) with `python requirements.py <expr-flag>` using the flags below

| Table/Figure | Content                                | Flag         | Estimated Time | Paper   |
| ------------ | -------------------------------------- | ------------ | -------------- | ------- |
| Table 3      | CF Example Quality - Random Forest     | `--tab3`     | ~10 mins       | Sec 7.3 |
| Table 4      | CF Example Quality - Gradient Boosting | `--tab4`     | ~15 mins       | Sec 7.4 |
| Figure 9     | Explanation Robustness                 | `--fig9`     | ~10 mins       | Sec 7.5 |
| Figure 10    | Explanation Personalization            | `--fig10`    | ~40 mins       | Sec 7.5 |
| Figure 11    | FACET's COREX Index Evaluation         | `--fig11`    | ~2 hrs 30 mins | Sec 7.6 |
| Figure 12    | FACET's COREX Index Evaluation         | `--fig12`    | ~3 hrs 15 mins | Sec 7.6 |
| Figure 13/14 | Model Scalability Evaluation           | `--fig13_14` | ~5 hrs 15 mins | Sec 7.7 |

After running the desired experiment(s) the results will be saved as CSVs in the [results/](/results/) directory

Figures and tables will automatically be generated and saved to the [figures/reproducibility/](/figures/reproducibility/) directory

## Runtime Notes

As the complete set of experiments for our paper took >100 hours to run, we take the following steps to save you time

- By default we run most experiments for only 1 iteration rather than 10
  - To disable this add the `--all_iterations` flag (roughly 10x the runtime)
- By default we skip running the INCREDIBLY slow MACE and RFOCSE comparison methods
  - To disable this add the `--all_baselines` flag
  - This will HUGELY increase runtime for experiments containing these methods (Table 3 and Figure 9)
- If you have sufficient excess hardware available you can safely launch multiple experiments in parallel using separate terminals
  - E.g., in one terminal `python requirements.py --tab3` and in another `python requirements.py --tab4`
  - This will save you time overall but may somewhat effect the accuracy of per sample explanation time metrics
- Adding multiple experiment flags in one execution is supported
  - E.g., `python requirement.py --fig11 --fig12 --all_iterations`
- Due to space limits in our paper we moved results for three dataset to [additional_results](/results/additional_results.md) here on GitHub
  - Figures and tables for these datasets are thus reproduced and saved separately as `<figname>_apdx` in reproduction
