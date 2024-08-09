# FACET: Robust Counterfactual Explanation Analytics

This repository provides FACET (Fast Actionable Counterfactuals for Ensembles of Trees) a method for generating counterfactual explanations of decisions made by tree ensembles. It is based on our papers *FACET: Robust Counterfactual Explanation Analytics* published at SIGMOD 2024 and a subsequent demo paper *Counterfactual Explanation Analytics: Empowering Lay Users to Take Action Against Consequential Automated Decisions* published at VLDB 2024. Short explainers, recorded talks, and open access to all our papers are available [here](https://petervannostrand.github.io/publications).

## FACET Overview

FACET generates a novel type of explanation which we call *counterfactual regions* for decisions made by ensembles of trees. For an instance `x` a counterfactual region `R` defines a portions of the feature space where all points `x' in R` are guaranteed to be counterfactual to `x`, e.g. if `y=f(x)=A` then `y=f(x')=B`. We design FACET to be highly performant and support a wide variety of user parameters such that explanations can be interactively personalized to meet real users needs.

## Installation

- This repository is validated on Python 3.10, [requirements.txt](./requirements.yml) contains a list of required packages formatted for use with Anaconda
- FACET's web user interface is built using JavaScript and uses the Node Package Manager to install and build all dependencies
- To run experiments with OCEAN, a state-of-the-art method we compare to, you will need a license for the Gurobi optimizer. Free academic licenses are available [here](https://www.gurobi.com/academia/academic-program-and-licenses/). Setup can be done as follows

Download and install [Anaconda](https://www.anaconda.com/download/success) and [Node.js](https://nodejs.org/en/download/prebuilt-installer), be sure to check `Automatically install the necessary tools` during Node installation

```bash
# create the anaconda environment
conda config --add channels https://conda.anaconda.org/gurobi
conda create --name facet --file requirements.txt
conda activate facet
# if running SOTA comparison method MACE, install required solver
pysmt-install --z3 --confirm-agreement
# if running SOTA comparison method OCEAN install and activate the gurobi optimizer
pip install --ignore-installed gurobipy
grbgetkey <your_acadmic_license_key>
# install JavaScript packages for the UI
cd webapp
npm install
```

Then continue with either running experiments or launching the Demo UI

## Running Experiments

For convenience [main.py](./main.py) takes a variety of command line arguments

| flag         | purpose                                              | allowed values                                                                         |
| ------------ | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `--expr`     | the experiment to run                                | `simple`, `ntrees`, `nrects`, `compare`, `k`, `m`, `nconstraints`, `perturb`, `robust` |
| `--values`   | the experimental values to test                      | space separated list of values e.g. `10 50 100` or `0.1 0.2 0.3`                       |
| `--ds`       | the dataset to explain                               | cancer, glass, magic, spambase, vertebral                                              |
| `--method`   | the XAI method to use                                | `FACETIndex`, `OCEAN`, `RFOCSE`, `AFT`, `MACE`                                         |
| `--ntrees`   | the ensemble size to test                            | integer value, overridden in for `--expr` `ntrees`                                     |
| `--maxdepth` | the max depth of ensemble trees                      | integer value, `-1` for no max depth                                                   |
| `--it`       | the iteration to run, used as random seed            | space separated integer values                                                         |
| `--fmod`     | a filename modifier append to append to results file | string value                                                                           |
| `--model`    | the underlying mode to explain                       | `rf` or `gbc`                                                                          |

Executing `python main.py` with no flags will perform a simple explanation of 20 instances on the vertebral dataset using FACET and an ensemble with `T=10, Dmax=5`. Parameters not involved in any given experiment are set to the default values provided in [experiments.py](./experiments/experiments.py)

All results are output to `./results/<expr_name>.csv`. Generated explanations, all parameters used in each iteration, and a summary of results can be found at `./results/<expr_name>.csv`. Code for generating all figures from the paper are available in Jupyter Notebooks at `./figures/<expr_name>.ipynb` and should be pointed to a matching results csv file of your choice.

## Demo UI

We develop a WebUI demonstrating FACET through a visual dashboard interface which we published at VLDB 2024's Demo Track

To launch this dashboard do the following

```bash
# launch the app
cd webapp
npm install
run run dev
```

Then navigate to `[http://localhost:5175/](http://localhost:5175/) in your browser

## Datasets

A listing of datasets which FACET has been applied on including the number of instance `N`, the number of features `n` and the number of features after one hot encoding. Datasets marked with `*` have results included in the paper with the remainder except `loans` presented here. All figures shown here and in the paper can be found in the `./figures/` directory.

| Dataset Name                                   | Abbreviated Name | N     | n   | n (one-hot) | Source                                                                                  |
| ---------------------------------------------- | ---------------- | ----- | --- | ----------- | --------------------------------------------------------------------------------------- |
| Adult*                                         | `adult`          | 45222 | 11  | 41          | [OCEAN](https://github.com/vidalt/OCEAN)                                                |
| Breast Cancer Wisconsin (Diagnostic) Data Set* | `cancer`         | 699   | 9   | 9           | [UCI](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) |
| ProPublica COMPAS Recidivism Data Set          | `compas`         | 5278  | 5   | 5           | [OCEAN](https://github.com/vidalt/OCEAN)                                                |
| Credit Card Default*                           | `credit`         | 29623 | 14  | 14          | [OCEAN](https://github.com/vidalt/OCEAN)                                                |
| Glass Identification Data Set                  | `glass`          | 214   | 9   | 9           | [UCI](https://archive.ics.uci.edu/ml/datasets/Glass+Identification)                     |
| Loan Predication (user study only)             | `loans`          | 615   | 13  | NA          | [Kaggle](https://www.kaggle.com/datasets/ninzaami/loan-predication)                     |
| MAGIC Gamma Telescope Data Set*                | `magic`          | 19020 | 10  | 10          | [UCI](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)                    |
| Spambase*                                      | `spambase`       | 4600  | 57  | 57          | [UCI](https://archive.ics.uci.edu/ml/datasets/Spambase)                                 |
| Vertebral Column Data Set                      | `vertebral`      | 310   | 6   | 6           | [UCI](https://archive.ics.uci.edu/ml/datasets/vertebral+column)                         |

## Academic Citations

Thank you for your interest in our work! Please use the following BibTeX citations when referencing FACET

### Core FACET Paper - SIGMOD 2024

```BibTeX
@article{10.1145/3626729,
    author = {VanNostrand, Peter M. and Zhang, Huayi and Hofmann, Dennis M. and Rundensteiner, Elke A.},
    title = {FACET: Robust Counterfactual Explanation Analytics},
    year = {2023},
    issue_date = {December 2023},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {1},
    number = {4},
    url = {https://doi.org/10.1145/3626729},
    doi = {10.1145/3626729},
    journal = {Proc. ACM Manag. Data},
    month = {dec},
    articleno = {242},
    numpages = {27},
}
```

### FACET Demo Paper - VLDB 2024

Develops FACET's explanation visual interface

```BibTeX
@article{10.14778/3685800.3685872,
    author = {VanNostrand, Peter M. and Hofmann, Dennis M. and Lei, Ma and Genin, Belisha and Huang, Randy and Rundensteiner, Elke A.},
    title = {Counterfactual Explanation Analytics: Empowering Lay Users to Take Action Against Consequential Automated Decisions},
    year = {2024},
    issue_date = {August 2024},
    publisher = {VLDB Endowment},
    volume = {17},
    number = {12},
    issn = {150-8097},
    url = {https://doi.org/10.14778/3685800.3685872},
    doi = {10.14778/3685800.3685872},
    journal = {Proc. VLDB Endow.},
    month = {aug},
    pages = {4349-4352},
    numpages = {4}
}
```
