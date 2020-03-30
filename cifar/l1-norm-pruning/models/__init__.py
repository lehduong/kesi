from __future__ import absolute_import

from .vgg import *
from .resnet import ResNet 
from .wrn import wrn
from .preresnet import preresnet

def resnet20(dataset='cifar10', cfg=None):
    return ResNet(depth=20, dataset=dataset, cfg=cfg)

def resnet32(dataset='cifar10', cfg=None):
    return ResNet(depth=32, dataset=dataset, cfg=cfg)

def resnet56(dataset='cifar10', cfg=None):
    return ResNet(depth=56, dataset=dataset, cfg=cfg)

def resnet110(dataset='cifar10', cfg=None):
    return ResNet(depth=110, dataset=dataset, cfg=cfg)
