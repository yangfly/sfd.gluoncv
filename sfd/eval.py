from __future__ import division
from __future__ import print_function

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata

import sys
sys.path.append('sfd')
from nn import SFD, get_sfd
from detector import SFDDetector
from data import WiderDetection
from data import WiderFaceEvalMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Eval SFD networks.')
    parser.add_argument('--network', '-n', type=str, default='vgg16',
                        help="Base network name")
    parser.add_argument('--use_bn', type=bool, default=False,
                        help="Whether enable base model to use batch-norm layer.")
    parser.add_argument('--model', '-m', type=str, default='',
                        help="Whether enable base model to use batch-norm layer.")
    parser.add_argument('--dataset', type=str, default='val',
                        help='Dataset to be evaluated.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Training with GPUs, you can specify 0 for example.')
    args = parser.parse_args()
    return args

def get_dataset(dataset):
    """Get dataset iterator"""
    assert dataset == 'val', 'evaluate only support val set'
    val_dataset = WiderDetection(splits='val')
    # val_metric = WiderFaceEvalMetric(iou_thresh=0.5, pbar=False)
    val_metric = WiderFaceEvalMetric(iou_thresh=0.5, pbar=True)
    return val_dataset, val_metric

def validate(detector, val_data, metric):
    """Test on validation dataset."""
    metric.reset()
    for img, label in val_data:
        # scores, bboxes = detector.detect(img)
        scores, bboxes = detector.ms_detect(img)
        metric.update(bboxes, scores, label)
    return metric.get()

def get_detector(name, use_bn, model, ctx):
    net = get_sfd(name, use_bn, model)
    net.input_reshape((6000, 1024))
    base = 1 if name.startswith('vgg') else 8
    return SFDDetector(net, base, ctx)

if __name__ == '__main__':
    args = parse_args()
    # context
    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    detector = get_detector(args.network, args.use_bn, args.model, ctx)

    # training data
    val_data, val_metric = get_dataset(args.dataset)

    # evaluation
    names, values = validate(detector, val_data, val_metric)
    for k, v in zip(names, values):
        print('{:7}MAP = {}'.format(k, v))
