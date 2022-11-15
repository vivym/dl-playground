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


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
    ):
        super().__init__()

        self.layers = nn.Sequential(*[
            BasicBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_layers)
        ])

        self.pooler = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(1, 3), padding=(0, 1),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)

        shortcut = x
        x = self.pooler(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = shortcut * x
        x = F.leaky_relu_(x)

        shortcut = x
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = F.leaky_relu_(x)

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
        CNNBlock(3, 64, 1),
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
