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
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]
COLORS = [[85.0, 0.0, 255.0], [0.0, 0.0, 255.0], [0.0, 85.0, 255.0], [0.0, 170.0, 255.0], [0.0, 255.0, 255.0], [0.0, 255.0, 170.0], [0.0, 255.0, 85.0], [0.0, 255.0, 0.0], [85.0, 255.0, 0.0], [170.0, 255.0, 0.0], [255.0, 255.0, 0.0], [255.0, 170.0, 0.0], [255.0, 85.0, 0.0], [255.0, 0.0, 0.0], [170.0, 0.0, 255.0], [255.0, 0.0, 170.0], [255.0, 0.0, 255.0], [255.0, 0.0, 85.0]]


def draw_line(img, traj, color, ratio, thickness=2, skip=1):
    for idx, (d1, d2) in enumerate(zip(traj[:-1], traj[1:])):
        if idx % skip != 0:
            continue
        cv2.line(img, (int(d1[0]*ratio), int(d1[1]*ratio)), (int(d2[0]*ratio), int(d2[1]*ratio)), color, thickness)
    return img

def draw_dotted_line(img, traj, color, ratio, thickness=2, r=0.85):
    for idx, (d1, d2) in enumerate(zip(traj[:-1], traj[1:])):
        x1 = int(d1[0] * r + d2[0] * (1 - r))
        y1 = int(d1[1] * r + d2[1] * (1 - r))
        x2 = int(d2[0] * r + d1[0] * (1 - r))
        y2 = int(d2[1] * r + d1[1] * (1 - r))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def draw_x(img, point, color, ratio, thickness=2):
    x, y = int(point[0]*ratio), int(point[1]*ratio)
    scale = thickness * 2
    cv2.line(img, (x - scale, y - scale), (x + scale, y + scale), color, thickness)
    cv2.line(img, (x + scale, y - scale), (x - scale, y + scale), color, thickness)
    return img


def draw_scale(img, cx, cy, rad, color):
    cv2.circle(img, (cx, cy), rad, color, 1)
    cv2.line(img, (cx, cy - rad), (cx, cy + rad), color, 1)
    cv2.line(img, (cx - rad, cy), (cx + rad, cy), color, 1)
    return img


def draw_pose(img, point, pose, scale, ratio):
    canvas = np.zeros_like(img)
    cx, cy = point
    pose = pose.reshape((18, 2))
    for idx, (x, y) in enumerate(pose):
        cv2.circle(canvas, (int((x*scale+cx)*ratio), int((y*scale+cy)*ratio)), int(scale / 7 * ratio), COLORS[idx], -1)
    for idx1, idx2 in PAIRS:
        line_color = list(map(lambda x: (x[0] + x[1]) / 2, zip(COLORS[idx1], COLORS[idx2])))
        x1, y1 = pose[idx1]
        x2, y2 = pose[idx2]
        cv2.line(canvas, (int((x1*scale+cx)*ratio), int((y1*scale+cy)*ratio)),
                 (int((x2*scale+cx)*ratio), int((y2*scale+cy)*ratio)), line_color, int(scale / 12 * ratio))

    return cv2.addWeighted(img, 1.0, canvas, 0.5, 0.0)
