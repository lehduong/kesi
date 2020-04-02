from __future__ import absolute_import

from .vgg import *
from .resnet import ResNet 
from .wrn import wrn
from .preresnet import preresnet

def vgg16(dataset='cifar10', cfg=None):
    return vgg(depth=16, dataset=dataset, cfg=cfg)

def resnet20(dataset='cifar10', cfg=None):
    return ResNet(depth=20, dataset=dataset, cfg=cfg)

def resnet32(dataset='cifar10', cfg=None):
    return ResNet(depth=32, dataset=dataset, cfg=cfg)

def resnet56(dataset='cifar10', cfg=None):
    return ResNet(depth=56, dataset=dataset, cfg=cfg)

def resnet110(dataset='cifar10', cfg=None):
    return ResNet(depth=110, dataset=dataset, cfg=cfg)

def preresnet110(dataset='cifar10', cfg=None, block_name='basicblock'):
    return preresnet(depth=110, dataset=dataset,cfg=cfg, block_name=block_name)

def wrn_28_10(dataset='cifar10', cfg=None):
    return wrn(depth=28,widen_factor=10,dataset=dataset,cfg=cfg)