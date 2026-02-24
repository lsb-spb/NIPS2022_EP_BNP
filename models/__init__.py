from .resnet import resnet20
import torch.nn as nn


def get_model(model, num_classes, norm_layer=nn.BatchNorm2d):
    if model == 'resnet20':
        return resnet20(num_classes, norm_layer=norm_layer)
    else:
        raise ValueError(f"Unknown model: {model}")