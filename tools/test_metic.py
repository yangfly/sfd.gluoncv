from __future__ import division
from __future__ import print_function

from tqdm import tqdm
from mxnet import gluon
import mxnet as mx
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.utils.bbox import bbox_xywh_to_xyxy
from ..utils import WiderDetection
from ..utils import WiderFaceMetric
from ..utils import SFDValTransform

import os
class PredBatchIter(object):
    def __init__(self, batch_size, txtfile, pred_dir):
        assert os.path.isfile(txtfile), 'expected file not existed. {}'.format(txtfile)
        assert os.path.isdir(pred_dir), 'expected dir not existed. {}'.format(pred_dir)
        self.batch_size = batch_size
        self._load_image_list(txtfile)
        self._load_pred(pred_dir)
        self.reset()
    
    def reset(self):
        self.current_id = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.num
    
    def next(self):
        begin = self.current_id
        end = min(begin + self.batch_size, self.num)
        if begin >= end:
            return None
        self.current_id = end
        mlen = 0
        for pred in self.preds[begin:end]:
            mlen = max(len(pred), mlen)
        batch = mx.nd.full((end-begin, mlen, 5), -1, dtype='float64')
        for i, pred in enumerate(self.preds[begin:end]):
            batch[i,:len(pred)] = pred
        return batch

    def _load_image_list(self, txtfile):
        # read image list from txtfile
        self.images = []
        with open(txtfile, 'r') as f:
            while True:
                image = f.readline().strip()
                if len(image) == 0:
                    break
                self.images.append(image)
                num = int(f.readline().strip())
                for i in range(num):
                    f.readline()
        self.num = len(self.images)
        
    def _load_pred(self, pred_dir):
        # read preds(bbox, score) from pred_dir
        self.preds = []
        for image in self.images:
            txtpath = os.path.join(pred_dir, image.rstrip('.jpg') + '.txt')
            with open(txtpath, 'r') as f:
                pred = map(lambda x : map(float, x.strip().split(' ')), f.readlines()[2:])
                pred = self.xywh_to_xyxy(pred)
            self.preds.append(pred)
    
    def xywh_to_xyxy(self, pred):
        _pred = []
        for rec in pred:
            rec[2] += 1 # inter-w to real-w
            rec[3] += 1 # inter-h to real-h
            rec = list(bbox_xywh_to_xyxy(rec[:4])) + [rec[4]]
            _pred.append(rec)
        return mx.nd.array(_pred, dtype='float64')

def get_dataset(iou_thresh, set_names):
    val_dataset = WiderDetection(splits='val')
    val_metric = WiderFaceMetric(iou_thresh=iou_thresh, set_names=set_names)
    return val_dataset, val_metric

def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SFDValTransform()), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='keep', num_workers=num_workers)
    return val_loader

def validate(preds, gts, metric):
    """Test on validation dataset."""
    metric.reset()
    with tqdm(total=len(preds)) as pbar:
        for gt in gts:
            pred = preds.next()
            det_bboxes = pred[:,:,:4]
            # clip to image size
            # det_bboxes.clip(gt[0].shape[2])
            det_scores = pred[:,:,4]
            gt_bboxes = gt[1]
            metric.update(det_bboxes, det_scores, gt_bboxes)
            pbar.update(det_bboxes.shape[0])
    return metric.get()

if __name__ == '__main__':
    # settings
    num_workers = 0
    batch_size = 100
    input_size = 640

    # val ground truth
    val_dataset, val_metric = get_dataset(0.5, ('easy', 'medium', 'hard'))
    val_data = get_dataloader(val_dataset, input_size, batch_size, num_workers)
    # val predictions
    txtfile = 'datasets/widerface/wider_face_split/wider_face_val_bbx_gt.txt'
    pred_dir = '../S3FD/SFD/sfd_test_code/WIDER_FACE/eval_tools_old-version/sfd_val'
    val_preds = PredBatchIter(batch_size, txtfile, pred_dir)
    names, values = validate(val_preds, val_data, val_metric)
    for k, v in zip(names, values):
        print(k, v)
