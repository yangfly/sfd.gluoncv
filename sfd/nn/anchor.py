# pylint: disable=unused-import
"""Anchor box generator for SFD detector."""
from __future__ import absolute_import

import numpy as np
from mxnet import gluon
from mxnet import nd
from gluoncv.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from gluoncv.nn.sampler import OHEMSampler, NaiveSampler
from gluoncv.nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from gluoncv.nn.bbox import BBoxCenterToCorner
from .matcher import CompensateMatcher

__all__ = ['SFDAnchorGenerator', 'SFDTargetGenerator']

class SFDAnchorGenerator(gluon.HybridBlock):
    """Bounding box anchor generator for SFD Face Detection.
    Adapted from gluoncv.data.ssd.anchor to support size_min alone

    Parameters
    ----------
    index : int
        Index of this generator in SSD models, this is required for naming.
    sizes : iterable of floats
        Sizes of anchor boxes.
    ratios : iterable of floats
        Aspect ratios of anchor boxes.
    step : int or float
        Step size of anchor boxes.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    offsets : tuple of float
        Center offsets of anchor boxes as (h, w) in range(0, 1).

    """
    def __init__(self, index, im_size, sizes, ratios, step, alloc_size=(128, 128),
                 offsets=(0.5, 0.5), clip=False, **kwargs):
        super(SFDAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        if isinstance(sizes, (list, tuple)):
            self._sizes = (sizes[0], np.sqrt(sizes[0] * sizes[1])) if len(sizes) > 1 else sizes
        else:
            self._sizes = [sizes]
        self._ratios = ratios
        self._step = step
        self._alloc_size = alloc_size
        self._offsets = offsets
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        self._key = 'anchor_%d' % (index)
        self.anchors = self.params.get_constant(self._key, anchors)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        """Generate anchors for once. Anchors are stored with (center_x, center_y, w, h) format."""
        assert len(sizes) >= 1, "SFD requires sizes at least min_size"
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step
                # ratio = ratios[0], size = size_min or sqrt(size_min * size_max)
                r = ratios[0]
                anchors.append([cx, cy, sizes[0], sizes[0]])
                if len(sizes) > 1:
                    anchors.append([cx, cy, sizes[1], sizes[1]])
                # size = sizes[0], ratio = ...
                for r in ratios[1:]:
                    sr = np.sqrt(r)
                    w = sizes[0] * sr
                    h = sizes[0] / sr
                    anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    def reset_anchors(self, alloc_size):
        self._alloc_size = alloc_size
        anchors = self._generate_anchors(self._sizes, self._ratios, self._step, self._alloc_size, self._offsets)
        self._reg_params.pop('anchors')
        self.anchors = gluon.Constant(self._key, anchors)
        self.params._params[self.prefix + self._key] = self.anchors
        self.anchors.initialize()

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._sizes) + len(self._ratios) - 1

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors):
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = F.concat(*[cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)], dim=-1)
        return a.reshape((1, -1, 4))


class SFDTargetGenerator(gluon.Block):
    """Training targets generator for SFD Face Detection.

    Parameters
    ----------
    iou_thresh : (float, float)
        IOU overlap threshold for compensate matching, default is (0.35, 0.1).
    topk: int
        Minimum number of match per label for compensate matching, default is 6
    neg_thresh : float
        IOU overlap threshold for negative mining, default is 0.5.
    negative_mining_ratio : float
        Ratio of hard vs positive for negative mining.
    stds : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.
    """
    def __init__(self, iou_thresh=(0.35, 0.1), topk=6, neg_thresh=0.5, negative_mining_ratio=3,
                 stds=(0.1, 0.1, 0.2, 0.2), **kwargs):
        super(SFDTargetGenerator, self).__init__(**kwargs)
        # self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(iou_thresh[0])])
        # self._matcher = MaximumMatcher(iou_thresh[0])
        self._matcher = CompensateMatcher(iou_thresh, topk)
        if negative_mining_ratio > 0:
            self._sampler = OHEMSampler(negative_mining_ratio, thresh=neg_thresh)
            self._use_negative_sampling = True
        else:
            self._sampler = NaiveSampler()
            self._use_negative_sampling = False
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)
        self._center_to_corner = BBoxCenterToCorner(split=False)

    # pylint: disable=arguments-differ
    def forward(self, anchors, cls_preds, gt_boxes, gt_ids):
        """Generate training targets."""
        anchors = self._center_to_corner(anchors.reshape((-1, 4)))
        ious = nd.transpose(nd.contrib.box_iou(anchors, gt_boxes), (1, 0, 2))
        matches = self._matcher(ious)
        if self._use_negative_sampling:
            samples = self._sampler(matches, cls_preds, ious)
        else:
            samples = self._sampler(matches)
        cls_targets = self._cls_encoder(samples, matches, gt_ids)
        box_targets, box_masks = self._box_encoder(samples, matches, anchors, gt_boxes)
        return cls_targets, box_targets, box_masks
