"""SFD Demo script."""
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import argparse
import mxnet as mx
import gluoncv as gcv
from matplotlib import pyplot as plt

import sys
sys.path.append('sfd')
from nn import SFD, get_sfd
from detector import SFDDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Test with SFD networks.')
    parser.add_argument('--network', '-n', type=str, default='vgg16',
                        help="Base network name")
    parser.add_argument('--use_bn', type=bool, default=False,
                        help="Whether enable base model to use batch-norm layer.")
    parser.add_argument('--model', '-m', type=str, default='',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--image', type=str, default='tools/selfie.jpg')
    parser.add_argument('--gpu', type=int, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    args = parser.parse_args()
    return args

def get_detector(name, use_bn, model, ctx):
    net = get_sfd(name, use_bn, model)
    net.input_reshape((6000, 2048))
    base = 1 if name.startswith('vgg') else 8
    return SFDDetector(net, base, ctx)

if __name__ == '__main__':
    args = parse_args()
    # context
    ctx = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()
    detector = get_detector(args.network, args.use_bn, args.model, ctx)
    img = mx.image.imread(args.image)
    scores, bboxes = detector.detect(img)
    # scores, bboxes = detector.ms_detect(img)
    ax = gcv.utils.viz.plot_bbox(img, bboxes, thresh=0)
    plt.show()
