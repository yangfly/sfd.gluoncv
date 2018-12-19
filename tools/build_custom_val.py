"""build custom val dataset for online validation."""

from __future__ import division
from __future__ import print_function

import os
import sys
import os.path as osp
import numpy as np
import mxnet as mx
import cv2 as cv
from scipy import io
from gluoncv.data.transforms import image as gimage
import gluoncv as gcv
from matplotlib import pyplot as plt
import random
random.seed(233)

def loadtxt(root, txt='wider_face_val_bbx_gt.txt'):
    print('Loading val annotations into memory...')
    anno_txt = osp.join(root, 'wider_face_split', txt)
    images = []
    labels = []
    with open(anno_txt, 'r') as f:
        while True:
            img = f.readline().strip()
            if img == '':
                break
            num = int(f.readline().strip())
            bbox = []
            for i in range(num):
                box = map(float, f.readline().strip().split()[:4])
                bbox.append(box)
            if len(bbox) > 0:
                images.append(img)
                labels.append(np.array(bbox, dtype=np.float32))
    return images, labels

def load_ground_truth(root, set_names=('easy', 'medium', 'hard')):
    """Load ground truth from mat files in `eval_tools`.
    
    Returns
    -------
    gt_lists : list
        Lists for faces idx in each set. 
    """
    gt_pattern = osp.join(root, 'eval_tools',
                         'ground_truth', 'wider_{}_val.mat')
    gt_lists = []
    for name in set_names:
        matfile = gt_pattern.format(name)
        _gt_lists = io.loadmat(matfile)['gt_list']
        gt_list = []
        for _gt_list in _gt_lists:
            for _gt in _gt_list[0]:
                gt_list.append(_gt[0].flatten() - 1)
        gt_lists.append(gt_list)
    return gt_lists

def imsave(filename, im):
    dir_name = osp.dirname(filename)
    if not osp.exists(dir_name):
        os.makedirs(dir_name)
    cv.imwrite(filename, im.asnumpy()[..., (2,1,0)])

def build_custom(size=640, root='widerface'):
    print('Building custom dataset...')
    images, labels = loadtxt(root)
    gt_lists = load_ground_truth(root)
    ipath = osp.join(root, 'WIDER_{}', 'images', '{}')
    txt = osp.join(root, 'wider_face_split', 'wider_face_custom_bbx_gt.txt')
    with open(txt, 'w') as f:
        for i, image, label, e, m, h in zip(range(len(images)), images, labels, *gt_lists):
            src = mx.image.imread(ipath.format('val', image))
            gt = np.zeros((label.shape[0], 3), dtype=label.dtype)
            gt[e, 0] = 1
            gt[m, 1] = 1
            gt[h, 2] = 1
            label = np.hstack((label, gt))
            im, bbox = transform(src, label, size)
            assert im is not None, 'transform failure: {}'.format(image)
            f.write('{}\n{}\n'.format(image, bbox.shape[0]))
            for box in bbox:
                f.write('{} {} {} {} {} {} {}\n'.format(*list(box.astype(int))))
            imsave(ipath.format('custom', image), im)
            # print(i, src.shape, im.shape)
            # print(bbox.astype(int))
            # bbox[:, 2:4] += bbox[:, :2]
            # ax = gcv.utils.viz.plot_bbox(im, bbox[:,:4])
            # plt.show()

def transform(src, label, size=640):
    # get im, bbox
    crop = try_crop(label, src.shape, size)
    if crop is None:
        crop = try_crop(label, src.shape, min(src.shape[:2]))
        if crop is None:
            crop = try_crop(label, src.shape, size, last_chance=True)
            if crop is None:
                crop = try_crop(label, src.shape, min(src.shape[:2]), last_chance=True)
    assert crop is not None, 'transform failure type: crop'
    bbox = bbox_crop(label, crop)
    if bbox.shape[0] == 0:
        bbox = np.zeros((1,7), dtype=np.float32)
    im = mx.image.fixed_crop(src, *crop)
    if crop[2] != size:
        im = gimage.imresize(im, size, size)
        bbox[:,:4] = bbox[:,:4] * (size / crop[2])
    return im, bbox
    
def bbox_crop(bbox, crop):
    # mask out bbox with center outside crop
    center = bbox[:, :2] + (bbox[:, 2:4] + 1) / 2
    mask = np.logical_and(crop[:2] <= center, center <= crop[:2] + crop[2:4]).all(axis=1)
    bbox = bbox[mask]
    # transform and clip borders
    bbox[:, 2:4] += bbox[:, :2] + 1
    bbox[:, :2] = np.maximum(bbox[:, :2], crop[:2])
    bbox[:, 2:4] = np.minimum(bbox[:, 2:4], crop[:2] + crop[2:])
    bbox[:, 2:4] -= bbox[:, :2] + 1
    bbox[:, :2] -= crop[:2]
    return bbox

def overlap(bbox, crop):
    tl = np.maximum(bbox[:, :2], crop[:2])
    br = np.minimum(bbox[:, :2] + bbox[:, 2:] + 1, crop[:2] + crop[2:])
    area_i = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
    area_a = np.prod(bbox[:, 2:] + 1, axis=1)
    return area_i / area_a

def try_crop(bbox, im_shape, size, max_trail=50, last_chance=False):
    size = min(size, min(im_shape[:2]))
    condidates = []
    nface = []
    h, w = im_shape[:2]
    gt = bbox[:,4:]
    bbox = bbox[:, :4]
    kbbox = bbox[(gt > 0).any(axis=1)]
    for i in range(max_trail):
        ct = random.randrange(h - size + 1)
        cl = random.randrange(w - size + 1)
        crop = np.array([cl, ct, size, size])
        ious = overlap(kbbox, crop)
        a = (ious > 0.99).sum()
        b = (ious < 0.01).sum()
        if a + b < kbbox.shape[0] and not last_chance:
            continue
        if b == 0:
            return crop
        condidates.append(crop)
        nface.append(a)
    if not condidates:
        return None
    else:
        return condidates[np.array(nface).argmax()]

if __name__ == '__main__':
    build_custom(640, 'widerface')