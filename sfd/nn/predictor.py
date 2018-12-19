"""Predictor for classification/box prediction."""
from __future__ import absolute_import
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from gluoncv.nn.predictor import ConvPredictor

class ConvMOPredictor(ConvPredictor):
    """Convolutional Max-out background classification predictor for conv3.
    ref: section 3.3 in https://arxiv.org/pdf/1708.05237.pdf
    It's useful to balance the positive vs.background samples from conv3_3 in sfd.

    Parameters
    ----------
    num_channel : int
        Number of conv channels.
    num_background : int
        Number of conv channels for background.
    kernel : tuple of (int, int), default (3, 3)
        Conv kernel size as (H, W).
    pad : tuple of (int, int), default (1, 1)
        Conv padding size as (H, W).
    stride : tuple of (int, int), default (1, 1)
        Conv stride size as (H, W).
    activation : str, optional
        Optional activation after conv, e.g. 'relu'.
    use_bias : bool
        Use bias in convolution. It is not necessary if BatchNorm is followed.

    """
    def __init__(self, num_channel, num_background=3, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                 activation=None, use_bias=True, **kwargs):
        super(ConvMOPredictor, self).__init__(num_channel, **kwargs)
        assert num_channel == 2, "Required num_channel = 2 but got {}".format(num_channel)
        assert num_background > 1, "Required background channel > 1 bug got {}".format(num_background)
        self.num_channel = num_channel
        self.num_background = num_background
        with self.name_scope():
            self.predictor = nn.Conv2D(num_channel + num_background - 1, kernel, 
                strides=stride, padding=pad, activation=activation, use_bias=use_bias, 
                weight_initializer=mx.init.Xavier(magnitude=2), bias_initializer='zeros')

    def hybrid_forward(self, F, x):
        x = self.predictor(x)
        fc = F.slice_axis(x, axis=1, begin=0, end=self.num_channel-1)
        bg = F.slice_axis(x, axis=1, begin=self.num_channel-1, end=None)
        bg = F.max_axis(bg, axis=1, keepdims=True)
        return F.concat(fc, bg, dim=1)