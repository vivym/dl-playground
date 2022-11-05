from typing import Optional, Any

from torchvision import models
from torchvision.models import SwinTransformer
from torchvision.models.swin_transformer import _swin_transformer


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


models.swin_m = swin_m
