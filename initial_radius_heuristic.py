import numpy as np
from dataset import load_data, DS_NAMES
import time


def cal_mean_min_interclass_dist(X_train, y_train):
  X_train_1, X_train_2 = X_train[y_train==0], X_train[y_train==1]
  dist = np.sum((X_train_1[:, None,:] - X_train_2)**2, axis=-1) ** 0.5
  min_dist_1 = np.min(dist, axis=1) #nn dist for class 1
  min_dist_2 = np.min(dist, axis=0) #nn dist for class 2
  d1 = np.sum(min_dist_1) / len(min_dist_1)
  d2 = np.sum(min_dist_2)/len(min_dist_2)
  avg_dist = (np.sum(min_dist_1) + np.sum(min_dist_2))/(len(min_dist_1)+len(min_dist_2))
  return d1, d2, avg_dist


def cal_mean_min_dist(X_train):
    dist = np.linalg.norm(X_train[:, None, :] - X_train[None, :, :], axis=-1)
    dist += np.eye(*dist.shape) * 2e31
    min_dist = np.min(dist, axis=1)
    mean_min_dist = np.mean(min_dist)
    return mean_min_dist

print("avearage nearest neighbot distances")

for ds in DS_NAMES:
    x, y = load_data(ds, preprocessing="Normalize")
    time_start = time.time()
    avg_dist = cal_mean_min_dist(x)
    time_end = time.time()
    runtime = time_end - time_start
    print("ds: {}, shape: {}, dist: {}, runtime: {}".format(ds, x.shape, avg_dist, runtime))

print("classwise avearage nearest neighbot distances")

for ds in DS_NAMES:
    x, y = load_data(ds, preprocessing="Normalize")
    time_start = time.time()
    d1, d2, avg_dist = cal_mean_min_interclass_dist(x, y)
    time_end = time.time()
    runtime = time_end - time_start
    print("ds: {}, shape: {}, d1: {}, d2: {}, avg_dust: {}, runtime: {}".format(ds, x.shape, d1, d2, avg_dist, runtime))