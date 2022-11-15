from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import SwinTransformer
from torchvision.models.swin_transformer import (
    _swin_transformer, SwinTransformerBlockV2, PatchMergingV2
)
from torchvision.models.resnet import BasicBlock


def swin_m(*, weights: Any, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_micro architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.
    """

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=64,
        depths=[2, 2, 4, 2],
        num_heads=[2, 4, 8, 16],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        weights=None,
        progress=progress,
        **kwargs,
    )


class CNN_part(nn.Module):
    def __init__(self, in_channel=3, channel=32):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channel, channel, 3, 1, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(channel, 2*channel, 3, 1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(2*channel, 2*channel, 3, 1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(2*channel, 2*channel, 3, 1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.conv3_1 = nn.Conv2d(2*channel, 2*channel, (1, 3), 1, padding=(0, 1), bias=False)
        self.relu3 = nn.LeakyReLU()

        self.conv4_1 = nn.Conv2d(2*channel, channel, 3, 1, padding=1, bias=False)
        self.relu4 = nn.LeakyReLU()
    def forward(self, x):
        org = x
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.relu1(x)
        res1 = x
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.relu2(x)
        x = x + res1

        x_tmp = x
        x = self.pool1(x)
        x = self.conv3_1(x)
        x = self.relu3(x)
        x = x*x_tmp

        x = self.conv4_1(x)
        x = self.relu4(x)
        x = torch.cat((x, org), dim=1)
        return x


def swin_t_conv1(*, weights: Any, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_tiny_conv architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.
    """

    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=64,
        depths=[2, 2, 4, 2],
        num_heads=[2, 4, 8, 16],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        weights=None,
        progress=progress,
        **kwargs,
    )

    """
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
    """

    # model.features[0].insert(1, CNNBlock(64, 64, 1))
    model.features[0][0] = nn.Sequential(
        CNN_part(3, 32),
        nn.Conv2d(
            64, 64, kernel_size=(4, 4), stride=(4, 4)
        ),
    )

    return model


def swin_v2_t_conv1(*, weights: Any, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_tiny_conv architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.
    """

    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )

    model.features[0].insert(1, CNNBlock(96, 96, 1))

    return model


def swin_v2_t_conv2(*, weights: Any, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_v2_tiny_conv architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.
    """

    model = _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )

    model.features[0].insert(1, CNNBlock(96, 96, 2))

    return model


models.swin_m = swin_m
models.swin_t_conv1 = swin_t_conv1
models.swin_v2_t_conv1 = swin_v2_t_conv1
models.swin_v2_t_conv2 = swin_v2_t_conv2
