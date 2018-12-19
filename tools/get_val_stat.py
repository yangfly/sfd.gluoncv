#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('.')
from tqdm import tqdm
from sfd import WiderDetection

dt = WiderDetection(splits='val')
H, W = 0, 0

with tqdm(total=len(dt)) as pbar:
    for im, _ in dt:
        h, w = im.shape[:2]
        H = max(H, h)
        W = max(W, w)
        pbar.update(1)
print('Max Image Size (HxW): {} x {}'.format(H, W))

