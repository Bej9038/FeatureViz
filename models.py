"""
''' models.py '''

Description: Functions to load pretrained torch models.
Author: Ben Jordan
"""
from torchvision import models
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


def vit():
    weights = models.ViT_B_32_Weights.IMAGENET1K_V1
    model = models.vit_b_32(weights=weights)
    model.requires_grad_(False)
    return model


def resnet():
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    model.requires_grad_(False)
    return model
