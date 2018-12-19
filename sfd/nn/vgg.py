"""Single-shot Scale-invariant Face Detector."""
from __future__ import absolute_import
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.initializer import Xavier
from gluoncv.model_zoo import vgg_atrous

__all__ = ['VGG16', 'VGG19']

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
        print("import base model params from {}".format(filename))
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

    def hybrid_forward(self, F, x, init_scale):
        raise NotImplementedError


class VGGExtractor(VGGBase):
    """VGG multi layer feature extractor which produces multiple output
    feauture maps.

    Parameters
    ----------
    layers : list of int
        Number of layer for vgg base network.
    filters : list of int
        Number of convolution filters for each layer.
    extras : dict of list
        Extra layers configurations.
    batch_norm : bool
        If `True`, will use BatchNorm layers.

    """
    def __init__(self, layers, filters, extras, batch_norm=False, **kwargs):
        super(VGGExtractor, self).__init__(layers, filters, batch_norm, **kwargs)
        with self.name_scope():
            self.extras = nn.HybridSequential()
            for i, config in enumerate(extras['conv']):
                extra = nn.HybridSequential(prefix='extra%d_'%(i))
                with extra.name_scope():
                    for f, k, s, p in config:
                        extra.add(nn.Conv2D(f, k, s, p, **self.init))
                        if batch_norm:
                            extra.add(nn.BatchNorm())
                        extra.add(nn.Activation('relu'))
                self.extras.add(extra)
        # normalize layer for 3/4/5-th stage
        self.norms = nn.HybridSequential()
        for f, init in zip(filters[2:], extras['normalize']):
            self.norms.add(vgg_atrous.Normalize(f, init))

    def hybrid_forward(self, F, x):
        assert len(self.stages) == 6
        outputs = []
        for stage in self.stages[:2]:
            x = stage(x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')
        for stage, norm in zip(self.stages[2:5], self.norms):
            x = stage(x)
            outputs.append(norm(x))
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')
        x = self.stages[5](x)
        outputs.append(x)
        for extra in self.extras:
            x = extra(x)
            outputs.append(x)
        return outputs


vgg_spec = {
    11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
    13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
    16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
    19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])
}

extra_spec = {
    'conv': [((256, 1, 1, 0), (512, 3, 2, 1)),  # conv6_
             ((128, 1, 1, 0), (256, 3, 2, 1))], # conv7_
    'normalize': (10, 8, 5),
}

def get_vgg_extractor(num_layers, pretrained=False, ctx=mx.cpu(),
                      root='models/pretrained', **kwargs):
    """Get VGG feature extractor networks.

    Parameters
    ----------
    num_layers : int
        VGG types, can be 11,13,16,19.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mx.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The returned network.

    """
    layers, filters = vgg_spec[num_layers]
    net = VGGExtractor(layers, filters, extra_spec, **kwargs)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        batch_norm_suffix = '_bn' if kwargs.get('batch_norm') else ''
        net.initialize(ctx=ctx)
        assert num_layers >= 16, "current import_params only support vgg 16 or 19, but got {}".format(num_layers)
        net.import_params(get_model_file('vgg%d%s'%(num_layers, batch_norm_suffix),
                                           tag=pretrained, root=root), ctx=ctx)
    return net

def VGG16(**kwargs):
    """Get VGG 16 layer feature extractor networks."""
    return get_vgg_extractor(16, **kwargs)

def VGG19(**kwargs):
    """Get VGG 19 layer feature extractor networks."""
    return get_vgg_extractor(19, **kwargs)

if __name__ == "__main__":
    net = VGG16(batch_norm=False, pretrained=True, ctx=mx.cpu())
    print(net)
    for key, param in net.collect_params().items():
        if param._data is not None:
            continue
        print(key)
        param.initialize()
    # net = VGG19(batch_norm=True, pretrained=True, ctx=mx.cpu())
    # print(net)
