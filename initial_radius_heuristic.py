import numpy as np
from dataset import load_data, DS_NAMES
import time


def cal_mean_min_dist(X_train):
    dist = np.linalg.norm(X_train[:, None, :] - X_train[None, :, :], axis=-1)
    dist += np.eye(*dist.shape) * 2e31
    min_dist = np.min(dist, axis=1)
    mean_min_dist = np.mean(min_dist)
    return mean_min_dist


for ds in DS_NAMES:
    x, y = load_data(ds, preprocessing="Normalize")
    time_start = time()
    avg_dist = cal_mean_min_dist(x)
    time_end = time()
    runtime = time_start - time_end
    print("ds: {}, shape: {}, dist: {}, runtime: {}".format(ds, x.shape, avg_dist, runtime))
