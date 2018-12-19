"""Single-shot Scale-invariant Face Detector."""
from __future__ import division
from __future__ import absolute_import
import numpy as np
import mxnet as mx

__all__ = ['SFDDetector']

class SFDDetector:
    def __init__(self, net, base=1, ctx=mx.cpu(0), 
                 mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        self.ctx = ctx
        self.base = base
        if callable(net):
            self.net = net
            self.net.collect_params().reset_ctx(ctx=ctx)
        else:
            raise ValueError("Required net as HybridBlock, but got {}".format(net))
        # self.set_nms()
    
    def set_nms(self, nms_thresh=0.3, nms_topk=5000, post_nms=750):
        self.net.set_nms(nms_thresh=nms_thresh, nms_topk=nms_topk, post_nms=post_nms)

    def detect(self, src, shrink=1):
        """Detect image with original size.
        src: mx.nd.ndarray 3-channels, RGB, [0,255], (h,w,3)
        shrink: shrink image and keep aspect ratio
        """
        dets = self._detect(src, shrink)
        bboxes = dets[:,:4]
        scores = dets[:,4]
        return scores, bboxes
    
    def _detect(self, src, shrink=1):
        assert shrink <= 1, 'shrink > 1 is not suitable.'
        h, w = src.shape[:2]
        nw = int(round(w * shrink / self.base)) * self.base
        nh = int(round(h * shrink / self.base)) * self.base
        im = mx.image.imresize(src, nw, nh)
        scale_x, scale_y = nw / w, nh / h
        # print(nw, nh, w, h, shrink)

        im = mx.nd.image.to_tensor(im)
        im = mx.nd.image.normalize(im, mean=self.mean, std=self.std)
        x = im.expand_dims(0).as_in_context(self.ctx)
        ids, scores, bboxes = [o[0].asnumpy() for o in self.net(x)]

        bboxes[:,(0,2)] /= scale_x
        bboxes[:,(1,3)] /= scale_y
        dets = np.hstack((bboxes, scores))
        return dets
    
    def _flip_detect(self, src, shrink=1):
        src_f = mx.nd.flip(src, axis=1)
        det_f = self._detect(src_f, shrink)
        width = src.shape[1]
        xmax = width - det_f[:, 0]
        xmin = width - det_f[:, 2]
        det_f[:, 0] = xmin
        det_f[:, 2] = xmax
        return det_f

    def ms_detect(self, src, shrinks=(0.5, 0.75)):
        dets = [self._detect(src, 1), self._flip_detect(src, 1)]
        dets.extend([self._detect(src, s) for s in shrinks])
        dets = np.row_stack(dets)
        dets = self._bbox_vote(dets)
        bboxes = dets[:,:4]
        scores = dets[:,4]
        return scores, bboxes
    
    def _bbox_vote(self, det, topk=750):
        order = det[:, 4].ravel().argsort()[::-1]
        det = det[order, :]
        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[0] + area[:] - inter)

            # get needed merge det and delete these det
            merge_index = np.where(o >= 0.3)[0]
            det_accu = det[merge_index, :]
            det = np.delete(det, merge_index, 0)

            if merge_index.shape[0] <= 1:
                continue
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

        dets = dets[:topk, :]
        return dets
