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
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--imdir_name', type=str, default="images")
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    start = time.time()

    video_id = os.path.basename(args.indir)
    data_dir = os.getenv("TRAJ_DATA_DIR")
    indir = os.path.join(data_dir, "videos", video_id)
    homography_path = os.path.join(data_dir, "homography/{}_homography.json".format(video_id))
    with open(homography_path, "r") as f:
        pose_dict = json.load(f)

    pose_path = os.path.join(data_dir, "pose/{}_pose.json".format(video_id))
    pose_dir = os.path.join(indir, "poses")
    vis_dir = os.path.join(indir, "images_pose_filtered")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    print(pose_dir)
    for input_path in sorted(glob.glob(os.path.join(pose_dir, "*.json"))):
        key = "rgb_{}.jpg".format(os.path.basename(input_path).split("_")[1])
        if key not in pose_dict:
            continue

        with open(input_path, "r") as f:
            data = json.load(f)

        raw_poses = [person["pose_keypoints"] for person in data["people"]]
        inside_bb_partial = partial(inside_bb, pose_dict[key], args.bb_threshold, args.score_threshold)

        pose_assign = filter(lambda x: x[0] is True, map(inside_bb_partial, raw_poses))
        poses = [list(chunked(x[1], 3)) for x in pose_assign]

        pose_dict[key]["pose"] = poses
        frame = int(key[4:9])
        if frame % 100 == 0:
            print(key)

        if not args.debug:
            continue

        # Debug: draw bb and pose
        img = cv2.imread(os.path.join(indir, args.imdir_name, key))
        for x1, y1, x2, y2, s in pose_dict[key]["detected"]:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2, 16)

        # Draw pose
        canvas = np.zeros_like(img)
        for ps in poses:
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

        out_fn = os.path.join(vis_dir, key)
        cv2.imwrite(out_fn, cv2.resize(result_img, None, fx=0.5, fy=0.5))

    if not os.path.exists(os.path.dirname(pose_path)):
        os.makedirs(os.path.dirname(pose_path))
    with open(pose_path, "w") as f:
        json.dump(pose_dict, f)

    print("Completed. Elapsed time: {} (s)".format(time.time()-start))
