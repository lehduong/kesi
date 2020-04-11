from __future__ import absolute_import

from .resnet import ResNet, BasicBlock, model_urls, model_zoo

def resnet34(pretrained=False, num_classes=200, cfg=None):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, cfg=cfg)
    if pretrained:
        state_dict_loaded = model_zoo.load_url(model_urls['resnet34'])
        model.load_state_dict(state_dict_loaded)

    return model

def resnet18(pretrained=False, num_classes=200, cfg=None):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cfg=cfg)
    if pretrained:
        state_dict_loaded = model_zoo.load_url(model_urls['resnet18'])
        model.load_state_dict(state_dict_loaded)

    return model