# FACET: Robust Counterfactual Explanation Analytics

This repository provides FACET (Fast Actionable Counterfactuals for Ensembles of Trees) a method for generating counterfactual explanations of decisions made by tree ensembles.
It is based on two related papers

- *[FACET: Robust Counterfactual Explanation Analytics](https://petervannostrand.github.io/publication/FACET-Robust-CFs)* published at SIGMOD 2024
- A subsequent demonstration paper *[Counterfactual Explanation Analytics: Empowering Lay Users to Take Action Against Consequential Automated Decisions](https://petervannostrand.github.io/publication/Examining-Actionable-Recourse)* published at VLDB 2024
- Short explainers, recorded talks, and free access to all our papers are available at [petervannostrand.github.io](https://petervannostrand.github.io/)

## FACET Overview

FACET generates a novel type of explanation which we call *counterfactual regions* for decisions made by ensembles of trees.
For an instance `x` a counterfactual region `R` defines a portions of the feature space where all points `x' in R` are guaranteed to be counterfactual to `x`,
e.g. if `y=f(x)=A` then `y=f(x')=B` for all points `x'` in `R`.
We design FACET to be highly performant and support a wide variety of user parameters such that explanations can be interactively personalized to meet real users needs
and provide a demonstration of these interactions through our [demo user interface](#demo-user-interface).

## Quick Start

1. Follow our [installation guide](/instructions/INSTALLATION.md) to install require dependencies
2. Run a quick experiment `python main.py --expr simple` or checkout one of the experiments below
3. Launch the FACET web app `cd webapp` > `npm install` > open [http://localhost:5175/](http://localhost:5175)

If you're reproducing the results of our paper see our comprehensive [reproducibility guide](/instructions/REPRODUCABILITY.md) for a dedicated workflow

## Running Experiments

For convenience [main.py](./main.py) takes a variety of command line arguments

| flag         | purpose                                              | allowed values                                                                            |
| ------------ | ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `--expr`     | the experiment to run                                | `simple`, `ntrees`, `nrects`, `compare`, `k`, `m`, `nconstraints`, `perturb`, `minrobust` |
| `--values`   | the experimental values to test                      | space separated list of values e.g. `10 50 100` or `0.1 0.2 0.3`                          |
| `--ds`       | the dataset to explain                               | a dataset name from the table below, e.g., `credit`                                       |
| `--method`   | the XAI method to use                                | `FACET`, `OCEAN`, `RFOCSE`, `AFT`, `MACE`                                                 |
| `--ntrees`   | the ensemble size to test                            | integer value, overridden for `--expr` `ntrees`                                           |
| `--maxdepth` | the max depth of ensemble trees                      | integer value, `-1` for no max depth                                                      |
| `--it`       | the iteration to run, used as random seed            | space separated integer values                                                            |
| `--fmod`     | a filename modifier append to append to results file | string value                                                                              |
| `--model`    | the underlying mode to explain                       | `rf` or `gbc`                                                                             |

Executing `python main.py` with no flags will perform a simple explanation of 20 instances on the vertebral dataset using FACET and an ensemble with `T=10, Dmax=5`. Parameters not involved in any given experiment are set to the default values provided in [experiments.py](./experiments/experiments.py)

All results are output to `./results/<expr_name>.csv`. Generated explanations, all parameters used in each iteration, and a summary of results can be found in the corresponding directory `./results/<expr-name>/`

Code for generating all figures from the paper are available in Jupyter Notebooks at `./figures/<expr_name>.ipynb` and should be pointed to a matching results csv file of your choice

## Demo User Interface

We develop a WebUI demonstrating FACET through a visual dashboard interface which we published at VLDB 2024's Demo Track

To launch this dashboard do the following

```bash
# launch the app
cd webapp
npm install
npm run dev
```

Then navigate to [http://localhost:5175/](http://localhost:5175/) in your browser

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
