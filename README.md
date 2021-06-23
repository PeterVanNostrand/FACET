# HEEAD

This report contains prototype code for the project *Heterogeneous Ensembles for Explainable Anomaly Detection* whose core principle is to use an ensemble of ensembles to perform anomaly detection in an explainable way. Explanations will be generated as contrastive examples, for example given a sample `x` which is classified `y=f(x)=-1` (an anomaly) the sample `x'` would be minimally modified version of `x` such that `y'!=y` so `y'=1` (an inlier).

## Requirements

The code in this repository was developed using Python 3.8.8 and uses the following python packages. Versions listed are used for consistency, however later versions will likely work

| Package       | Version |
|---------------|---------|
| Numpy         |  1.20.1 |
| Matplotlib    |   3.3.4 |
| Scikit learn  |  0.24.1 |
| Scipy         |   1.6.2 |
| Pandas        |   1.2.4 |
| H5PY          |  2.10.0 |

## Data Sources

| Dataset Name | Abbreviated Name | # Points | # Dimensions | # Outliers | Source | Features |
|--------------|------------------|----------|--------------|------------|--------|----------|
| Annthyroid | `annthyroid` | 7200 | 6 | 534 (7.42%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/annthyroid-dataset/) | real |
| Cardiotocogrpahy Dataset | `cardio` | 1831 | 21 | 176 (9.6%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/) | real |
| HTTP (KDDCUP99) | `http` | 567479 | 3 | 2211 (0.4%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/http-kddcup99-dataset/) | TODO |
| Mulcross | `mulcross` | 262144 | 4 | 26214 (10%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/) | TODO Not available on ODDS, sourced as csv from [OpenML](https://www.openml.org/d/40897) |
| Musk | `musk` | 3062 | 166 | 97 (3.2%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/musk-dataset/) | integer |
| Pendigits | `pendigits` | 6870 | 16 | 156 (2.27%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/pendigits-dataset/) | integer |
| Satimage-2 | `satimage` | 5803 | 36 | 71 (1.2%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/satimage-2-dataset/) | real |
| Shuttle | `shuttle` | 49097 | 9 | 3511 (7%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/shuttle-dataset/) | integer |
| Thyroid Disease Dataset | `thyroid` | 3772 | 6 | 93 (2.5%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/) | real |
| Wisconsin-Breast Cancer Diagnostics Dataset | `wbc` | 278 | 30 | 21 (5.6%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/wbc/) | real |
## Data Sources Not in Use

| Dataset Name | Abbreviated Name | # Points | # Dimensions | # Outliers | Source | Notes |
|--------------|------------------|----------|--------------|------------|--------|----------|
| Pima | `pima` | 768 | 8 | 268 (35%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/pima-indians-diabetes-dataset/) | [Pulled by dataset owner?](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes) |
| PageBlock | | | | | | Have to find |
| SpamBase | | | | | | Have to find |