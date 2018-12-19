"""Single-shot Scale-invariant Face Detector."""
from __future__ import absolute_import
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.initializer import Xavier

class VGGBase(gluon.HybridBlock):
    """VGG multi layer base network. You must inherit from it to define
    how the features are computed.

    Parameters
    ----------
    layers : list of int
        Number of layer for vgg base network.
    filters : list of int
        Number of convolution filters for each layer.
    batch_norm : bool, default is False
        If `True`, will use BatchNorm layers.

    """
    def __init__(self, layers, filters, batch_norm=False, **kwargs):
        super(VGGBase, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        self.init = {
            'weight_initializer': Xavier(
                rnd_type='gaussian', factor_type='out', magnitude=2),
            'bias_initializer': 'zeros'
        }
        self.layers = layers
        self.batch_norm = batch_norm
        with self.name_scope():
            self.stages = nn.HybridSequential()
            for l, f in zip(layers, filters):
                stage = nn.HybridSequential(prefix='')
                with stage.name_scope():
                    for _ in range(l):
                        stage.add(nn.Conv2D(f, kernel_size=3, padding=1, **self.init))
                        if batch_norm:
                            stage.add(nn.BatchNorm())
                        stage.add(nn.Activation('relu'))
                self.stages.add(stage)

            # use convolution instead of dense layers
            stage = nn.HybridSequential(prefix='fc_')
            with stage.name_scope():
                stage.add(nn.Conv2D(1024, kernel_size=3, padding=1, **self.init)) # fc6
                if batch_norm:
                    stage.add(nn.BatchNorm())
                stage.add(nn.Activation('relu'))
                stage.add(nn.Conv2D(1024, kernel_size=1, **self.init)) # fc7
                if batch_norm:
                    stage.add(nn.BatchNorm())
                stage.add(nn.Activation('relu'))
            self.stages.add(stage)

    def import_params(self, filename, ctx):
        """Import parameters from vgg16 classification model"""
        loaded = mx.nd.load(filename)
        params = self._collect_params_with_prefix()
        i = 0
        for k, l in enumerate(self.layers):
            j = 0
            for _ in range(l):
                # conv param
                for suffix in ['weight', 'bias']:
                    key_i = '.'.join(('features', str(i), suffix))
                    key_j = '.'.join(('stages', str(k), str(j), suffix))
                    assert key_i in loaded, "Params '{}' missing in {}".format(key_i, loaded.keys())
                    assert key_j in params, "Params '{}' missing in {}".format(key_j, params.keys())
                    params[key_j]._load_init(loaded[key_i], ctx)
                i += 1
                j += 1
                if self.batch_norm:
                    # batch norm param
                    for suffix in ['beta', 'gamma', 'running_mean', 'running_var']:
                        key_i = '.'.join(('features', str(i), suffix))
                        key_j = '.'.join(('stages', str(k), str(j), suffix))
                        assert key_i in loaded, "Params '{}' missing in {}".format(key_i, loaded.keys())
                        assert key_j in params, "Params '{}' missing in {}".format(key_j, params.keys())
                        params[key_j]._load_init(loaded[key_i], ctx)
                    i += 1
                    j += 1
                i += 1
                j += 1
            i += 1
        
        # stage 5
        params['stages.5.0.weight']._load_init(loaded['features.%d.weight' % i].reshape(4096,512,7,7)[:1024,:,2:5,2:5], ctx)
        params['stages.5.0.bias']._load_init(loaded['features.%d.bias' % i][:1024], ctx)
        i += 2
        j = 3 if self.batch_norm else 2
        params['stages.5.%d.weight' % j]._load_init(loaded['features.%d.weight' % i][:1024,:1024].reshape(1024,1024,1,1), ctx)
        params['stages.5.%d.bias' % j]._load_init(loaded['features.%d.bias' % i][:1024], ctx)

    def hybrid_forward(self, F, x):
        # raise NotImplementedError
        assert len(self.stages) == 6
        outputs = []
        for stage in self.stages[:5]:
            x = stage(x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')
        # x = self.stages[5](x)
        return x

class VGG(gluon.HybridBlock):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each feature block.
    filters : list of int
        Numbers of filters in each feature block. List length should match the layers.
    classes : int, default 1000
        Number of classification classes.
    batch_norm : bool, default False
        Use batch normalization.
    """
    def __init__(self, layers, filters, classes=1000, batch_norm=False, **kwargs):
        super(VGG, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm)

    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        # x = self.output(x)
        return x

vgg_spec = {
    11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
    13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
    16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
    19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])
}

from gluoncv.data.transforms.presets.imagenet import transform_eval
from mxnet import nd, image
from gluoncv.model_zoo.model_store import get_model_file
from numpy.testing import assert_array_almost_equal 

if __name__ == "__main__":
    ctx = mx.gpu(0)
    input_pic = '../gluon-cv/street.jpg'
    img = image.imread(input_pic)
    img = transform_eval(img).as_in_context(ctx)
    print(img.shape)

    for num_layers in [16, 19]:
        for bn in [False, True]:
            _bn = '_bn' if bn else ''
            name = 'vgg{}{}'.format(num_layers, _bn)
            model = get_model_file(name, tag=True, root='models')
            print('--------------- {} -------------'.format(model))
            layers, filters = vgg_spec[num_layers]
            vgg = VGG(layers, filters, batch_norm=bn)
            vgg.initialize(ctx=ctx)
            vgg.load_parameters(model, ctx, ignore_extra=True)
            vggbase = VGGBase(layers, filters, batch_norm=bn)
            vggbase.initialize(ctx=ctx)
            vggbase.import_params(model, ctx)
            pred1 = vgg(img)
            pred2 = vggbase(img)
            print('pred1 shape:', pred1.shape)
            print('pred2 shape:', pred2.shape)
            assert_array_almost_equal(pred1.asnumpy(), pred2.asnumpy())