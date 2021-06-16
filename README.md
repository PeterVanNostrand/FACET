# HEEAD

This report contains prototype code for the project *Heterogeneous Ensembles for Explainable Anomaly Detection* whose core principle is to use an ensemble of ensembles to perform anomaly detection in an explainable way. Explanations will be generated as contrastive examples, for example given a sample `x` which is classified `y=f(x)=-1` (an anomaly) the sample `x'` would be minimally modified version of `x` such that `y'!=y` so `y'=1` (an inlier).

## Requirements

The code in this repository was developed using Python 3.8.8 and uses the following python packages

- Numpy 1.20.1
- Matplotlib 3.3.4
- Scikit-learn 0.24.1
- Scipy 1.6.2
- TQDM 4.59.0
- Pandas 1.2.4

## Data Sources

| Dataset Name | Abbreviated Name | # Points | # Dimensions | # Outliers | Source |
|--------------|------------------|----------|--------------|------------|--------|
| Thyroid Disease Dataset | `thyroid` | 3772 | 6 | 93 (2.5%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/) |
| Wisconsin-Breast Cancer Diagnostics Dataset | `wbc` | 278 | 30 | 21 (5.6%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/wbc/) |
| Cardiotocogrpahy Dataset | `cardio` | 1831 | 21 | 176 (9.6%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/) |
| Musk | `musk` | 3062 | 166 | 97 (3.2%) | [StonyBrook ODDS](http://odds.cs.stonybrook.edu/musk-dataset/) |
