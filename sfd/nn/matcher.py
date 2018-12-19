from __future__ import absolute_import
import mxnet as mx
import numpy as np
from mxnet import gluon
import mobula

__all__ = ['CompensateMatcher']

class CompensateMatcher(gluon.HybridBlock):
    """A Matcher implementing  Scale compensation anchor matching strategy. (batch supported)

    Parameters
    ----------
    iou_thresh : (float, float)
        IOU overlap threshold, default is (0.35, 0.1).
    limit: int
        Minimum number of match per label, default is 6
    """
    def __init__(self, iou_thresh=(0.35, 0.1), topk=6):
        super(CompensateMatcher, self).__init__()
        self._thre1 = iou_thresh[0]
        self._thre2 = iou_thresh[1]
        self._topk = topk
        mobula.op.load('sfd/nn/matcher')
    
    def forward(self, x):
        gtids = x.argmax(axis=-1).astype(np.int32)
        ious = x.pick(gtids, axis=-1)
        match = mx.nd.empty_like(ious)
        mobula.func.compensate(x.shape[0], gtids, ious, match,
                               x.shape[1], x.shape[2],
                               self._thre1, self._thre2, self._topk)
        return match
    
    # def forward(self, ious):
    #     # from pudb import set_trace
    #     # set_trace()
    #     argmax = ious.argmax(axis=-1)
    #     amax = ious.pick(argmax, axis=-1).asnumpy()
    #     argmax = argmax.asnumpy().astype(np.int)
    #     labelnum = ious.shape[-1]
    #     if argmax.ndim == 2: # batch
    #         match = np.empty_like(argmax)
    #         for i in range(match.shape[0]):
    #             match[i,:] = self.compensate(amax[i], argmax[i], labelnum)
    #     elif argmax.ndim == 1:
    #         match = self.compensate(amax, argmax, labelnum)
    #     else:
    #         raise ValueError("Invalid shape of ious, ndim = {}".format(ious.ndim))
    #     return nd.array(match, dtype=np.float32)
    
    # def compensate(self, amax, argmax, labelnum):
    #     argmax[amax < self._thre2] = -1
    #     argids, ids, _ = zip(*sorted(
    #         zip(argmax, range(len(amax)), amax), key=lambda x:x[2], reverse=True))
    #     argids = np.array(argids)
    #     ids = np.array(ids)
    #     match = np.where(amax>=self._thre1, argmax, -1)
    #     for i in range(labelnum):
    #         if (match==i).sum() < self._limit:
    #             match[ids[argids==i][:self._limit]] = i
    #     return match