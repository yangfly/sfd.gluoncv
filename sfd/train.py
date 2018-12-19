"""Train SFD"""
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import argparse
import sys
import logging
logging.basicConfig(format="[%(asctime)s] %(message)s")
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.utils.metrics.accuracy import Accuracy

sys.path.append('sfd')
from nn import SFD, get_sfd
from data import WiderDetection
from data import WiderFaceMetric
from data import SFDTrainTransform, SFDValTransform

def parse_args():
    parser = argparse.ArgumentParser(description='Train SFD networks.')
    parser.add_argument('--network', type=str, default='vgg16',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--use_bn', type=bool, default=False,
                        help="Whether enable base model to use batch-norm layer.")
    parser.add_argument('--data-shape', type=int, default=640,
                        help="Input data shape, use 640.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='train',
                        help='Training dataset. Now support train, train,val.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=24, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./sfd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epoches at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='models/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--match-high-thresh', type=float, default=0.35,
                        help='High threshold for anchor matching.')
    parser.add_argument('--match-low-thresh', type=float, default=0.1,
                        help='Low threshold for anchor matching.')
    parser.add_argument('--match-topk', type=int, default=6,
                        help='Topk for anchor matching.')
    args = parser.parse_args()
    return args

def get_dataset(dataset):
    """Get dataset iterator"""
    if dataset == 'train,val':
        dataset = ('train', 'val')
    else:
        assert dataset == 'train', "Invalid training dataset: {}".format(dataset)
    train_dataset = WiderDetection(splits=dataset)
    val_dataset = WiderDetection(splits='custom')
    val_metric = WiderFaceMetric(iou_thresh=0.5)
    return train_dataset, val_dataset, val_metric

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader: transform and batchify."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    # stack image, cls_targets, box_targets
    batchify_fn = Tuple(Stack(), Stack(), Stack())
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(
            SFDTrainTransform(width, height, anchors, (args.match_high_thresh, args.match_low_thresh), args.match_topk)),
        batch_size, shuffle=True, batchify_fn=batchify_fn, last_batch='rollover', 
    num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SFDValTransform()),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def save_params(net, best_map, current_map, maps, epoch, save_interval, prefix):
    def update_symlink(src, dest):
        try:
            os.remove(dest)
        except:
            pass
        os.symlink(src, dest)

    current_map = float(current_map)
    model_path = '{:s}_{:03d}.params'.format(prefix, epoch)
    best_path = '{:s}_best.params'.format(prefix)
    net.save_parameters(model_path)
    with open(prefix+'_maps.log', 'a') as f:
        msg = '{:03d}:\t{:.4f}\t  {:.6f} {:.6f} {:.6f}'.format(epoch, current_map, *maps)
        if current_map > best_map[0]:
            best_map[0] = current_map
            update_symlink(model_path, best_path)
            f.write(msg + '    *\n')
        else:
            f.write(msg + '\n')

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    # net.set_nms(nms_thresh=0.3, nms_topk=5000, post_nms=750)
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_scores = []
        gt_bboxes = []
        gt_lists = []
        for x, y in zip(data, label):
            _, scores, bboxes = net(x)
            det_scores.append(scores)
            det_bboxes.append(bboxes)
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_lists.append(y.slice_axis(axis=-1, begin=4, end=7))
        # update metric
        eval_metric.update(det_bboxes, det_scores, gt_bboxes, gt_lists)
    return eval_metric.get()

def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(log_file_path) and args.start_epoch == 0:
        os.remove(log_file_path)
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs+1):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize()
        # logger.info('1: {:.3f}'.format(time.time()-tic)) #########
        for i, batch in enumerate(train_data):
            # logger.info('2 {:.3f}'.format(time.time()-tic)) #########
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            # logger.info('3 {:.3f}'.format(time.time()-tic)) #########
            with autograd.record():
                cls_preds = []
                box_preds = []
                for x in data:
                    cls_pred, box_pred, _ = net(x)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2))
        
        # if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
        val_steps = [40, 20, 10, 5, 1]
        # if epoch % val_steps[epoch/50] == 0: # for normal
        if epoch % 5 == 0:  # for run, fast
            # consider reduce the frequency of validation to save time
            vtic = time.time()
            names, maps = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{:7}MAP = {}'.format(k, v) for k, v in zip(names, maps)])
            logger.info('[Epoch {}] Validation: {:.3f}\n{}'.format(epoch, (time.time()-vtic), val_msg))
            current_map = sum(maps) / len(maps)
            save_params(net, best_map, current_map, maps, epoch, args.save_interval, args.save_prefix)
        else:
            current_map = 0.

if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net = get_sfd(args.network, args.use_bn, pretrained=args.resume)
    network = args.network + ('_bn' if args.use_bn else '')
    args.save_prefix = os.path.join(args.save_prefix, network, 'sfd')

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
