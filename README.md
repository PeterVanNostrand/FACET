# HEEAD

This report contains prototype code for the project *Heterogeneous Ensembles for Explainable Anomaly Detection* whose core principle is to use an ensemble of ensembles to perform anomaly detection in an explainable way. Explanations will be generated as contrastive examples, for example given a sample `x` which is classified `y=f(x)=-1` (an anomaly) the sample `x'` would be minimally modified version of `x` such that `y'!=y` so `y'=1` (an inlier).
