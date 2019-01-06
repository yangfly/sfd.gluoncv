import mxnet as mx
import numpy as np
import random

import mobula
mobula.op.load('sfd/nn/matcher')

def compensate(amax, argmax, labelnum, thre1, thre2, topk):
    argmax[amax < thre2] = -1
    argids, ids, _ = zip(*sorted(
        zip(argmax, range(len(amax)), amax), key=lambda x:x[2], reverse=True))
    argids = np.array(argids)
    ids = np.array(ids)
    match = np.where(amax>=thre1, argmax, -1)
    for i in range(labelnum):
        if (match==i).sum() < topk:
            match[ids[argids==i][:topk]] = i
    return match

def test():
    ious = [0.52, 0.01, 0.30, 0.35,
            0.01, 0.34, 0.26, 0.44,
            0.63, 0.32, 0.47, 0.35,
            0.23, 0.45, 0.31, 0.12,
            0.03, 0.21, 0.34, 0.02]
    gtids = np.arange(20, dtype=np.int) / 4
    merge = zip(ious, gtids)
    random.shuffle(merge)
    ious, gtids = zip(*merge)
    print(ious)
    print(gtids)
    match1 = compensate(np.array(ious, dtype=np.float32), np.array(gtids, dtype=np.int),
                        5, 0.35, 0.1, 2)
    print(match1)
    mious = mx.nd.array([ious], dtype=np.float32)
    mgtids = mx.nd.array([gtids], dtype=np.int32)
    match2 = mx.nd.zeros_like(mious)
    mobula.func.compensate(1, mgtids, mious, match2, len(gtids), 5, 0.35, 0.1, 2)
    print(match2[0])

test()
