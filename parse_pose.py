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
import argparse
import json
import time
import glob
import cPickle as pickle

from functools import partial
from more_itertools import chunked
import numpy as np
import cv2

UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15]
PART_NAME = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "Background"]
PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
COLORS = [[85.0, 0.0, 255.0], [0.0, 0.0, 255.0], [0.0, 85.0, 255.0], [0.0, 170.0, 255.0], [0.0, 255.0, 255.0], [0.0, 255.0, 170.0], [0.0, 255.0, 85.0], [0.0, 255.0, 0.0], [85.0, 255.0, 0.0], [170.0, 255.0, 0.0], [255.0, 255.0, 0.0], [255.0, 170.0, 0.0], [255.0, 85.0, 0.0], [255.0, 0.0, 0.0], [170.0, 0.0, 255.0], [255.0, 0.0, 170.0], [255.0, 0.0, 255.0], [255.0, 0.0, 85.0]]


if __name__ == "__main__":
    """
    Parse raw result from Openpose to our format
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_threshold', type=float, default=0.05)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    start = time.time()
    data_dir = "minipose_annotation"
    vis_dir = "plots"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    X, Y, S = [], [], []
    for input_path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(input_path, "r") as f:
            data = json.load(f)

        raw_poses = [person["pose_keypoints"] for person in data["people"]]

        poses = np.array([list(chunked(x, 3)) for x in raw_poses])[0]
        spine = (poses[8:9, :2] + poses[11:12, :2]) / 2
        neck = poses[1:2, :2]
        sizes = np.linalg.norm(neck - spine, axis=1)  # (N, T, 1)
        sizes[sizes == 0] = 1e-8  # Avoid ZerodivisionError
        pose_normalized = (poses[...,:2] - spine) / sizes[:, np.newaxis]  # Normalization
        poses[..., :2] = pose_normalized
        # print(poses)
        pid = int(os.path.basename(input_path).split("_")[0][-3:])

        X.append(poses)
        Y.append((pid % 5) * 0.25)
        S.append(sizes[0])

        impath = os.path.join("minipose", os.path.basename(input_path).split("_")[0]+".jpg")
        img = cv2.imread(impath)

        # Draw pose
        canvas = np.zeros_like(img)
        for ps in poses[np.newaxis, ...]:
            if ps[1][0] == 0.0 or ps[8][0] == 0.0:
                continue
            scale = np.sqrt((ps[8][0] - ps[1][0]) ** 2 + (ps[8][1] - ps[1][1]) ** 2)
            for idx, (x, y, s) in enumerate(ps):
                if s > args.score_threshold:
                    cv2.circle(canvas, (int(x), int(y)), int(scale / 10), COLORS[idx], -1)
            for idx1, idx2 in PAIRS:
                if ps[idx1][2] >= args.score_threshold and ps[idx2][2] >= args.score_threshold:
                    cv2.line(img, (int(ps[idx1][0]), int(ps[idx1][1])), (int(ps[idx2][0]), int(ps[idx2][1])),
                             list(map(lambda x: (x[0] + x[1]) / 2, zip(COLORS[idx1], COLORS[idx2]))), int(scale / 18))

        result_img = cv2.addWeighted(img, 1.0, canvas, 0.5, 0.0)

        out_fn = os.path.join(vis_dir, os.path.basename(impath))
        cv2.imwrite(out_fn, cv2.resize(result_img, None, fx=0.5, fy=0.5))

    with open("data.pkl", "w") as f:
        pickle.dump({"X": X, "Y": Y, "S": S}, f)
    print(np.array(X).shape)
    print("Completed. Elapsed time: {} (s)".format(time.time()-start))
