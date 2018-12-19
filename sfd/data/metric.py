"""Wider Face Detection evaluation."""
from __future__ import division

import os
from collections import defaultdict
import numpy as np
import mxnet as mx
from scipy import io
from tqdm import tqdm

__all__ = ['WiderFaceMetric', 'WiderFaceEvalMetric']

class WiderFaceMetric(mx.metric.EvalMetric):
    """
    Mean average precision metric for online validation

    Parameters:
    ---------
    iou_thresh : float, default 0.5
        IOU overlap threshold for TP
    """
    SETS = ('easy', 'medium', 'hard')

    def __init__(self, iou_thresh=0.5, set_names=('easy', 'medium', 'hard')):
        super(WiderFaceMetric, self).__init__('WiderFaceMeanAP')
        self.set_names = set_names
        self.num = len(self.set_names)
        self.iou_thresh = iou_thresh
        self.reset()
    
    @property
    def sets(self):
        """Metric sets."""
        return self.set_names
    
    def reset(self):
        """Clear the collectors and internal statistics to initial state."""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        self._n_pos = defaultdict(int)
        self._score = defaultdict(list)
        self._match = defaultdict(list)
        self.pred_bbox_collector = []
        self.pred_score_collector = []
        self.gt_bbox_collector = []
        self.gt_list_collector = []
    
    def _norm_score(self):
        """Normalize predicted scores to [0, 1]."""
        max_score = 0.
        min_score = 1.
        for score in self.pred_score_collector:
            if len(score) > 0:
                max_score = max(max_score, np.max(score))
                min_score = min(min_score, np.min(score))
        diff = max_score - min_score
        for score in self.pred_score_collector:
            score[...] = (score - min_score) / diff

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        self._update()  # update metric at this time
        if self.num is None:
            if self.num_inst == 0:
                return (self.set_names, float('nan'))
            else:
                return (self.set_names, self.sum_metric / self.num_inst)
        else:
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (self.set_names, values)

    # pylint: disable=arguments-differ, too-many-nested-blocks
    def update(self, pred_bboxes, pred_scores, gt_bboxes, gt_lists):
        """Update internal buffer with latest prediction and gt pairs.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where N is the number of bboxes.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        gt_bboxes : mxnet.NDArray or numpy.ndarray
            Ground-truth bounding boxes with shape `B, M, 4`.
            Where M is the number of ground-truths.
        gt_lists : mxnet.NDArray or numpy.ndarray
            Ground-truth mask with shape `B, M, 3`.
            Where M is the number of ground-truths.
        """
        def as_numpy(a):
            """Convert a (list of) mx.NDArray into numpy.ndarray"""
            if isinstance(a, (list, tuple)):
                out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
                out = np.array(out)
                return np.concatenate(out, axis=0)
            elif isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a
        
        for pred_bbox, pred_score, gt_bbox, gt_list in zip(
                *[as_numpy(x) for x in [pred_bboxes, pred_scores, gt_bboxes, gt_lists]]):
            # strip padding -1 for pred and gt
            valid_pred = np.where(pred_score.flat >= 0)[0]
            self.pred_score_collector.append(pred_score.flat[valid_pred])
            self.pred_bbox_collector.append(pred_bbox[valid_pred, :])
            valid_gt = np.where(gt_bbox[:,0].flat >= 0)[0]
            self.gt_bbox_collector.append(gt_bbox[valid_gt, :])
            self.gt_list_collector.append(gt_list[valid_gt, :])
        
    def _eval_internal(self):
        """Compute internal statistics for full-set collectors."""
        self._norm_score()

        for pred_bbox, pred_score, gt_bbox, gt_lists in zip(self.pred_bbox_collector,
                self.pred_score_collector, self.gt_bbox_collector, self.gt_list_collector):
            for l in range(self.num):

                # sort by score // might not
                order = pred_score.argsort()[::-1]
                pred_bbox = pred_bbox[order]
                pred_score = pred_score[order]

                # update internal statistics
                gt_list = np.where(gt_lists[:, l] > 0.5)[0]
                self._n_pos[l] += len(gt_list)
                self._score[l].extend(pred_score)

                if len(pred_score) == 0:
                    continue
                if len(gt_bbox) == 0:
                    self._match[l].extend((0,)*len(pred_score))
                    continue
                
                # VOC evaluation follows integer typed bounding boxes.
                iou = bbox_overlaps(pred_bbox, gt_bbox)
                gt_index = iou.argmax(axis=1)
                gt_index[iou.max(axis=1) < self.iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_idx not in gt_list:
                            self._match[l].append(-1)
                        else:
                            if not selec[gt_idx]:
                                selec[gt_idx] = True
                                self._match[l].append(1)
                            else:
                                self._match[l].append(0)         
                    else:
                        self._match[l].append(0)

    def _update(self):
        """ update num_inst and sum_metric """
        aps = []
        self._eval_internal()
        recall, precs = self._recall_prec()
        for l, rec, prec in zip(range(len(precs)), recall, precs):
            ap = self._average_precision(rec, prec)
            aps.append(ap)
            if self.num is not None and l < self.num:
                self.sum_metric[l] = ap
                self.num_inst[l] = 1

    def _recall_prec(self, nthread=1000):
        """ get recall and precision from internal records """
        prec = np.zeros((self.num, nthread))
        rec = np.zeros((self.num, nthread))

        for l in range(self.num):
            score_l = np.array(self._score[l])
            match_l = np.array(self._match[l], dtype=np.int32)
            order = score_l.argsort()[::-1]
            score_l = score_l[order]
            match_l = match_l[order]

            for t in range(nthread):
                thresh = 1 - float(t+1) / nthread
                r_index = np.where(score_l >= thresh)[0]
                if len(r_index) > 0:
                    match_t = match_l[:r_index[-1]+1]
                    tp = float(np.sum(match_t == 1))
                    fp = float(np.sum(match_t == 0))
                    # If an element of fp + tp is 0,
                    # the corresponding element of prec[l,t] is nan.
                    prec[l,t] = tp / (fp + tp) if tp+fp!= 0 else float('nan')
                    # If n_pos[l] is 0, rec[l,t] is None.
                    rec[l,t] = tp / self._n_pos[l] if self._n_pos[l] != 0 else float('nan')

        return rec, prec

    def _average_precision(self, rec, prec):
        """
        calculate average precision

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        if rec is None or prec is None:
            return np.nan

        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], np.nan_to_num(prec), [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

def bbox_overlaps(bbox_a, bbox_b):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    ! differenct between Fast R-CNN bbox_overlaps and gluon-cv bbox_iou

    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.

    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4]) + 1

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + 1, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + 1, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


class WiderFaceEvalMetric(WiderFaceMetric):
    """Mean average precision metric for offline evaluation

    Parameters:
    ---------
    root : str, default 'widerface'
        Path to folder storing the eval_tools.
    iou_thresh : float, default 0.5
        IOU overlap threshold for TP
    set_names : list of str, default ('easy', 'medium', 'hard')
        optional, if provided, will print out AP for each class
    """
    SETS = ('easy', 'medium', 'hard')

    def __init__(self, root='widerface', iou_thresh=0.5, pbar=False,
                 set_names=('easy', 'medium', 'hard')):
        self.pbar = False
        super(WiderFaceEvalMetric, self).__init__(iou_thresh, set_names)
        assert len(set_names) > 0, "get 0, at least one set is required."
        for name in set_names:
            assert name in type(self).SETS, "set name {} invalid.".format(name)
        self.num = len(set_names)
        self.set_names = set_names
        self.gt_pattern = os.path.join(root, 'eval_tools',
                                      'ground_truth', 'wider_{}_val.mat')
        self.gt_lists = self._load_ground_truth()
        self.im_num = len(self.gt_lists[0])
        self.pbar = pbar
        self.reset()
    
    def _load_ground_truth(self):
        """Load ground truth from mat files in `eval_tools`.
        
        Returns
        -------
        gt_lists : list
           Lists for faces idx in each set. 
        """
        gt_lists = []
        for name in self.set_names:
            matfile = self.gt_pattern.format(name)
            _gt_lists = io.loadmat(matfile)['gt_list']
            gt_list = []
            for _gt_list in _gt_lists:
                for _gt in _gt_list[0]:
                    gt_list.append(_gt[0].flatten() - 1)
            gt_lists.append(gt_list)
        return gt_lists
    
    def reset(self):
        super(WiderFaceEvalMetric, self).reset()
        if self.pbar:
            self.pbar = tqdm(total=self.im_num)
    
    # pylint: disable=arguments-differ, too-many-nested-blocks
    def update(self, pred_bboxes, pred_scores, gt_bboxes):
        """Update internal buffer with latest prediction and gt pairs.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `N, 4`.
            Where N is the number of bboxes.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `N`.
        gt_bboxes : mxnet.NDArray or numpy.ndarray
            Ground-truth bounding boxes with shape `M, 4`.
            Where M is the number of ground-truths.
        """
        # strip padding -1 for pred and gt
        valid_pred = np.where(pred_scores.flat >= 0)[0]
        self.pred_score_collector.append(pred_scores.flat[valid_pred])
        self.pred_bbox_collector.append(pred_bboxes[valid_pred, :])
        valid_gt = np.where(gt_bboxes[:,0].flat >= 0)[0]
        self.gt_bbox_collector.append(gt_bboxes[valid_gt, :])
        if self.pbar: self.pbar.update(1)
    
    def _eval_internal(self):
        """Compute internal statistics for full-set collectors."""
        assert len(self.gt_bbox_collector) == self.im_num, \
                "only full-set evaluation is support: ({} < {})".format(
                    len(self.gt_bbox_collector), self.im_num)

        self._norm_score()
        for l in range(self.num):
            for pred_bbox, pred_score, gt_bbox, gt_list in zip(self.pred_bbox_collector,
                    self.pred_score_collector, self.gt_bbox_collector, self.gt_lists[l]):

                # sort by score // might not
                order = pred_score.argsort()[::-1]
                pred_bbox = pred_bbox[order]
                pred_score = pred_score[order]

                # update internal statistics
                self._n_pos[l] += len(gt_list)
                self._score[l].extend(pred_score)

                if len(pred_score) == 0:
                    continue
                if len(gt_bbox) == 0:
                    self._match[l].extend((0,)*len(pred_score))
                    continue
                
                # VOC evaluation follows integer typed bounding boxes.
                iou = bbox_overlaps(pred_bbox, gt_bbox)
                gt_index = iou.argmax(axis=1)
                gt_index[iou.max(axis=1) < self.iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_idx not in gt_list:
                            self._match[l].append(-1)
                        else:
                            if not selec[gt_idx]:
                                selec[gt_idx] = True
                                self._match[l].append(1)
                            else:
                                self._match[l].append(0)         
                    else:
                        self._match[l].append(0)

