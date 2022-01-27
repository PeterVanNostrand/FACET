# FACET

This repository contains prototype code for the project *Fast Actionable Counterfactual Examples for Ensembles of Trees* which generates simple human understandable explanations of the decisions made by a random forest model. Explanations are generated as counterfactual examples, for example given a sample `x` which is classified `y=f(x)=A` the sample `x'` would be minimally modified version of `x` such that `y'!=y` so `y'=B`.

## Requirements

The code in this repository was developed using Python 3.8.8 and uses the following python packages. Versions listed are used for consistency, however later versions will likely work

| Package      | Version |
| ------------ | ------- |
| H5PY         | 2.10.0  |
| Matplotlib   | 3.3.4   |
| Networkx     | 2.6.2   |
| Numpy        | 1.20.1  |
| Pandas       | 1.2.4   |
| Scikit-learn | 0.24.1  |
| Scipy        | 1.6.2   |

## Data Sources

| Dataset Name                                  | Abbreviated Name | # Points | # Dimensions | nClass                   | Source                                                                                  | Features      |
| --------------------------------------------- | ---------------- | -------- | ------------ | ------------------------ | --------------------------------------------------------------------------------------- | ------------- |
| Breast Cancer Wisconsin (Diagnostic) Data Set | `cancer`         | 699      | 9            | 2                        | [UCI](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) | real          |
| Glass Identification Data Set                 | `glass`          | 214      | 9            | 6 (2 float vs non-float) | [UCI](https://archive.ics.uci.edu/ml/datasets/Glass+Identification)                     | real          |
| MAGIC Gamma Telescope Data Set                | `magic`          | 19020    | 10           | 2                        | [UCI](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)                    | real          |
| Spambase                                      | `spambase`       | 4600     | 57           | 2                        | [UCI](https://archive.ics.uci.edu/ml/datasets/Spambase)                                 | real, integer |
| Vertebral Column Data Set                     | `vertebral`      | 310      | 6            | 2                        | [UCI](https://archive.ics.uci.edu/ml/datasets/vertebral+column)                         | real          |

## Anomaly Detection Data Sources

| Dataset Name                                | Abbreviated Name | # Points | # Dimensions | # Outliers  | Source                                                                     | Features |
| ------------------------------------------- | ---------------- | -------- | ------------ | ----------- | -------------------------------------------------------------------------- | -------- |
| Annthyroid                                  | `annthyroid`     | 7200     | 6            | 534 (7.42%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/annthyroid-dataset/)       | real     |
| Cardiotocogrpahy Dataset                    | `cardio`         | 1831     | 21           | 176 (9.6%)  | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/) | real     |
| HTTP (KDDCUP99)                             | `http`           | 567479   | 3            | 2211 (0.4%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/http-kddcup99-dataset/)    | real     |
| Mulcross                                    | `mulcross`       | 262144   | 4            | 26214 (10%) | [OpenML](https://www.openml.org/d/40897)                                   | real     |
| Musk                                        | `musk`           | 3062     | 166          | 97 (3.2%)   | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/musk-dataset/)             | integer  |
| Pendigits                                   | `pendigits`      | 6870     | 16           | 156 (2.27%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/pendigits-dataset/)        | integer  |
| Satimage-2                                  | `satimage`       | 5803     | 36           | 71 (1.2%)   | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/satimage-2-dataset/)       | real     |
| Shuttle                                     | `shuttle`        | 49097    | 9            | 3511 (7%)   | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/shuttle-dataset/)          | integer  |
| Thyroid Disease Dataset                     | `thyroid`        | 3772     | 6            | 93 (2.5%)   | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/)  | real     |
| Wisconsin-Breast Cancer Diagnostics Dataset | `wbc`            | 278      | 30           | 21 (5.6%)   | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/wbc/)                      | real     |

## Data Sources Not in Use

| Dataset Name | Abbreviated Name | # Points | # Dimensions | # Outliers | Source                                                                          | Notes                                                                                     |
| ------------ | ---------------- | -------- | ------------ | ---------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Pima         | `pima`           | 768      | 8            | 268 (35%)  | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/pima-indians-diabetes-dataset/) | [Pulled by dataset owner?](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes) |
| PageBlock    |                  |          |              |            |                                                                                 | Have to find                                                                              |
| SpamBase     |                  |          |              |            |                                                                                 | Have to find                                                                              |
