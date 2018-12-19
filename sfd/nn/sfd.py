"""Single-shot Scale-invariant Face Detector."""
from __future__ import absolute_import

import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock
from gluoncv.nn.predictor import ConvPredictor
from gluoncv.nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from data import WiderDetection
from .vgg import VGG16, VGG19
from .anchor import SFDAnchorGenerator
from .feature import FeatureExpander
from .predictor import ConvMOPredictor

# import sys
# sys.path.append('sfd/data')
# from dataset import WiderDetection
# sys.path.append('sfd/nn')
# from vgg import VGG16, VGG19
# from feature import FeatureExpander
# from predictor import ConvMOPredictor
# sys.path.append('sfd')
# from nn.anchor import SFDAnchorGenerator


__all__ = ['SFD', 'get_sfd']

# model configures
_models = {
    'vgg16'        :  (None, VGG16, 64),
    'vgg19'        :  (None, VGG19, 64),
    'resnet18'     :  ('resnet18_v1', ['stage1_activation1', 'stage2_activation1', 'stage3_activation1', 'stage4_activation1'], 64),
    'resnet34'     :  ('resnet34_v1', ['stage1_activation2', 'stage2_activation3', 'stage3_activation1', 'stage4_activation2'], 64),
    'resnet50'     :  ('resnet50_v1', ['stage1_activation2', 'stage2_activation3', 'stage3_activation1', 'stage4_activation2'], 64),
    'mobilenet'    :  ('mobilenet1.0', ['relu6_fwd', 'relu10_fwd', 'relu22_fwd', 'relu26_fwd'], 128),
    'mobilenet.75' :  ('mobilenet0.75', ['relu6_fwd', 'relu10_fwd', 'relu22_fwd', 'relu26_fwd'], 96),
    'mobilenet.5'  :  ('mobilenet0.5', ['relu6_fwd', 'relu10_fwd', 'relu22_fwd', 'relu26_fwd'], 64),
}

class SFD(HybridBlock):
    """Single-shot Scale-invariant Face Detector: https://arxiv.org/pdf/1708.05237.

    Parameters
    ----------
    network : string or None
        Name of the base network, if `None` is used, will instantiate the
        base network from `features` directly instead of composing.
    base_size : int
        Base input size, it is speficied so SFD can support dynamic input shapes.
    features : list of str or mxnet.gluon.HybridBlock
        Intermediate features to be extracted or a network with multi-output.
        If `network` is `None`, `features` is expected to be a multi-output network.
    num_filters : list of int
        Number of channels for the appended layers, ignored if `network`is `None`.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SFD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SFD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of all categories.
    use_1x1_transition : bool
        Whether to use 1x1 convolution as transition layer between attached layers,
        it is effective reducing model capacity.
    use_bn : bool
        Whether to use BatchNorm layer after each attached convolutional layer.
    reduce_ratio : float
        Channel reduce ratio (0, 1) of the transition layer.
    min_depth : int
        Minimum channels for the transition layers.
    global_pool : bool
        Whether to attach a global average pooling layer as the last output layer.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    stds : tuple of float, default is (0.1, 0.1, 0.2, 0.2)
        Std values to be divided/multiplied to box encoded values.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    ctx : mx.Context
        Network context.

    """
    def __init__(self, network, base_size, features, num_filters, sizes, ratios,
                 steps, classes, use_1x1_transition=True, use_bn=False, fpn_channel=64,
                 reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.35, nms_topk=5000, post_nms=750,
                 ctx=mx.cpu(), **kwargs):
        super(SFD, self).__init__(**kwargs)
        if network is None:
            num_layers = len(steps)
        else:
            num_layers = len(features) + len(num_filters) + int(global_pool)
        assert isinstance(sizes, list), "Must provide sizes as list or list of list"
        assert isinstance(ratios, list), "Must provide ratios as list or list of list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = [ratios] * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
                num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "SFD require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.base_size = base_size
        self.im_size = [base_size, base_size]

        with self.name_scope():
            if network is None:
                # use fine-grained manually designed block as features
                self.features = features(batch_norm=use_bn, pretrained=pretrained, ctx=ctx)
            else:
                self.features = FeatureExpander(
                    network=network, outputs=features, num_filters=num_filters,
                    use_1x1_transition=use_1x1_transition, fpn_channel=fpn_channel,
                    use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                    global_pool=global_pool, pretrained=pretrained, ctx=ctx)
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            asz = [base_size // 4, base_size // 4]
            for i, s, r, st in zip(range(num_layers), sizes, ratios, steps):
                anchor_generator = SFDAnchorGenerator(i, self.im_size, s, r, st, asz)
                self.anchor_generators.add(anchor_generator)
                # asz = max(asz // 2, 16)  # pre-compute larger than 16x16 anchor map
                asz = [max(sz // 2, 16) for sz in asz]
                num_anchors = anchor_generator.num_depth
                cls_num_channel = num_anchors * (len(self.classes) + 1)
                if i == 0:
                    self.class_predictors.add(ConvMOPredictor(cls_num_channel, 3))
                else:
                    self.class_predictors.add(ConvPredictor(cls_num_channel))
                self.box_predictors.add(ConvPredictor(num_anchors * 4))
            self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)
    
    def input_reshape(self, im_size=(1024, 1024)):
        assert min(im_size) >= self.base_size
        self.im_size = im_size
        asz = [sz // 4 for sz in im_size]
        for ag in self.anchor_generators:
            ag.reset_anchors(asz)
            asz = [max(sz // 2, 16) for sz in asz]

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    def set_nms(self, nms_thresh=0.3, nms_topk=5000, post_nms=750):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.3.
            Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
        nms_topk : int, default is 5000
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 750
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        features = self.features(x)
        cls_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.class_predictors)]
        box_preds = [F.flatten(F.transpose(bp(feat), (0, 2, 3, 1)))
                     for feat, bp in zip(features, self.box_predictors)]
        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features, self.anchor_generators)]
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))
        if autograd.is_training():
            return [cls_preds, box_preds, anchors]
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes

    def reset_class(self, classes):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.

        """
        self._clear_cached_op()
        self.classes = classes
        # replace class predictors
        with self.name_scope():
            class_predictors = nn.HybridSequential(prefix=self.class_predictors.prefix)
            for i, ag in zip(range(len(self.class_predictors)), self.anchor_generators):
                prefix = self.class_predictors[i].prefix
                new_cp = ConvPredictor(ag.num_depth * (self.num_classes + 1), prefix=prefix)
                new_cp.collect_params().initialize()
                class_predictors.add(new_cp)
            self.class_predictors = class_predictors
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

def get_sfd(network, use_bn=False, pretrained=False, **kwargs):
    """Get SFD models"""
    name, features, fpn_channel = _models[network]
    pretrained_base = False if pretrained else True
    net = SFD(name, 640, features, num_filters=[512, 256], reduce_ratio=0.5,
              sizes=[16, 32, 64, 128, 256, 512], ratios=[1], fpn_channel=fpn_channel,
              steps=[4, 8, 16, 32, 64, 128], classes=WiderDetection.CLASSES,
              pretrained=pretrained_base, use_bn=use_bn, **kwargs)
    if pretrained:
        assert isinstance(pretrained, str), "pretrained represents path to pretrained model."
        net.load_parameters(pretrained)
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    return net

# def sfd_vgg16(base_size=640, use_bn=False, pretrained=
#               pretrained=False, pretrained_base=True, **kwargs):
#     """SFD architecture with VGG16 base network for Widerface.

#     Parameters
#     ----------
#     pretrained : bool or str
#         Boolean value controls whether to load the default pretrained weights for model.
#         String value represents the path of pretrained model.
#     pretrained_base : bool or str, optional, default is True
#         Load pretrained base network, the extra layers are randomized.

#     Returns
#     -------
#     HybridBlock
#         A SSD detection network.
#     """
#     classes = WiderDetection.CLASSES
#     pretrained_base = False if pretrained else pretrained_base
#     net = SFD(None, 640, VGG16, None,
#               sizes=[16, 32, 64, 128, 256, 512], ratios=[1],
#               steps=[4, 8, 16, 32, 64, 128],
#               pretrained=pretrained_base, use_bn=use_bn,
#               classes=classes, ctx=mx.cpu(), **kwargs)
#     if pretrained:
#         assert isinstance(pretrained, str), "pretrained represents path to pretrained model."
#         net.load_parameters(pretrained, ctx=ctx)
#     return net

# def sfd_vgg19(base_size=640, use_bn=False, 
#               pretrained=False, pretrained_base=True, **kwargs):
#     """SFD architecture with VGG16 base network for Widerface.

#     Parameters
#     ----------
#     pretrained : bool or str
#         Boolean value controls whether to load the default pretrained weights for model.
#         String value represents the path of pretrained model.
#     pretrained_base : bool or str, optional, default is True
#         Load pretrained base network, the extra layers are randomized.

#     Returns
#     -------
#     HybridBlock
#         A SSD detection network.
#     """
#     classes = WiderDetection.CLASSES
#     pretrained_base = False if pretrained else pretrained_base
#     net = SFD(None, 640, VGG19, None,
#               sizes=[16, 32, 64, 128, 256, 512], ratios=[1],
#               steps=[4, 8, 16, 32, 64, 128],
#               pretrained=pretrained_base, use_bn=use_bn,
#               classes=classes, ctx=mx.cpu(), **kwargs)
#     if pretrained:
#         assert isinstance(pretrained, str), "pretrained represents path to pretrained model."
#         net.load_parameters(pretrained, ctx=ctx)
#     return net

if __name__ == '__main__':
    import sys
    network = sys.argv[1]
    use_bn = len(sys.argv) > 2
    net = get_sfd(network, use_bn)
    print(net.features)

    x = mx.sym.var('data')
    sym = mx.sym.Group(net.features(x))
    # mx.viz.print_summary(net, {'data': (1,3, 640, 640)})
    mx.viz.plot_network(sym, shape={'data': (1,3, 640, 640)}).render('net', cleanup=True)

    # for key, param in net.collect_params().items():
    #     if param._data is not None:
    #         continue
    #     print(key)
    #     param.initialize()