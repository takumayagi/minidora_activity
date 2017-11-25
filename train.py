#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

from __future__ import print_function
from __future__ import division
from six.moves import range

import os
import numpy as np
import cPickle as pickle

import cv2

PART_NAME = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "Background"]

with open("data.pkl") as f:
    data_dict= pickle.load(f)

X = np.array(data_dict["X"])
Y = np.array(data_dict["Y"])
S = np.array(data_dict["S"])
names = data_dict["names"]
pids = [name[:-3] for name in names]
X = X[:, [2, 3, 4, 5, 6, 7], :]  # 足だけ
X = X[...,:2].reshape((len(X), -1))

lmbdas = [0.0, 0.001, 0.01, 0.1, 0.2, 0.4]  # Regularization coefficient
print(np.unique(names))

# Cross-subject evalutation
for test_pid in np.unique(pids):
    test_idxs = [idx for idx, pid in enumerate(pids) if pid == test_pid]
    train_idxs = [idx for idx in range(len(pids)) if idx not in test_idxs]
    tr_X = X[train_idxs]
    tr_Y = Y[train_idxs]
    ts_X = X[test_idxs]
    ts_Y = Y[test_idxs]
    best_lmbda, best_score = -1, np.finfo(np.float64).max
    for idx, lmbda in enumerate(lmbdas):
        # Linear regression (with L2 regularization term)
        theta = np.linalg.solve(np.dot(tr_X.T, tr_X) + lmbda * np.eye(tr_X.shape[1]), np.dot(tr_X.T, tr_Y))
        err = np.mean(np.abs(ts_Y - np.dot(ts_X, theta)))
        if err < best_score:
            best_score = err
            best_lmbda = lmbda

    # Pick best score and plot to prediction
    theta = np.linalg.solve(np.dot(tr_X.T, tr_X) + best_lmbda * np.eye(tr_X.shape[1]), np.dot(tr_X.T, tr_Y))
    print(test_pid, best_lmbda, best_score)
    curr_idx = 0
    for name in [x for idx, x in enumerate(PART_NAME) if idx in [2, 3, 4, 5, 6, 7]]:
        for corr in ["X", "Y"]:
            print("{} {} {}".format(name, corr, theta[curr_idx]))
            curr_idx += 1

    prediction = np.dot(ts_X, theta)
    for idx, pred, gt in zip(test_idxs, prediction, ts_Y):
        impath = os.path.join("minipose_annotation", "{}_rendered.png".format(names[idx]))
        img = cv2.imread(impath)
        cv2.putText(img, "{:.2f} {:.2f}".format(gt, pred), (20, 100), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 0), 2)
        cv2.imwrite("results/{}.jpg".format(names[idx]), img)
