from typing import Optional, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchmetrics import Accuracy


class TorchvisionWrapper(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        in_channels: int = 3,
        num_classes: int = 1000,
        pretrained: bool = False,
        learning_rate: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        norm_weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        max_epochs: int = 60,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert hasattr(models, model_name)

        model_factory = getattr(models, model_name)

        if pretrained:
            weights = models._api._get_enum_from_fn(model_factory).DEFAULT
        else:
            weights = None
        self.model = model_factory(weights=weights, num_classes=num_classes)

        if in_channels != 3:
            if "swin" in model_name:
                conv = self.model.features[0][0]
            else:
                conv = next(self.model.children())
                assert isinstance(conv, nn.Conv2d)
            weight = conv.weight
            conv.weight = nn.Parameter(
                torch.empty(
                    weight.shape[0], in_channels, *weight.shape[2:],
                    dtype=weight.dtype, device=weight.device,
                )
            )
            conv.reset_parameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.norm_weight_decay = norm_weight_decay
        self.label_smoothing = label_smoothing
        self.max_epochs = max_epochs

        self.accuracy_1 = Accuracy(num_classes=num_classes, top_k=1)
        if num_classes > 5:
            self.accuracy_5 = Accuracy(num_classes=num_classes, top_k=5)

    def forward(self, x):
        return self.model(x)

    def _training_and_validation_step(self, batch, batch_idx: int):
        images, labels = batch

        outputs = self.forward(images)

        loss = F.cross_entropy(
            outputs, labels,
            label_smoothing=self.label_smoothing
        )

        acc1 = self.accuracy_1(outputs, labels) * 100
        if self.num_classes > 5:
            acc5 = self.accuracy_5(outputs, labels) * 100
        else:
            acc5 = None

        return loss, acc1, acc5

    def training_step(self, batch, batch_idx: int):
        loss, acc1, acc5 = self._training_and_validation_step(batch, batch)

        self.log("train/loss", loss, on_step=True)
        self.log(
            "train/acc@1", acc1,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        if self.num_classes > 5:
            self.log(
                "train/acc@5", acc5,
                on_step=True, on_epoch=True, prog_bar=True,
            )

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, acc1, acc5 = self._training_and_validation_step(batch, batch)

        self.log("val/loss", loss, on_step=True)
        self.log(
            "val/acc@1", acc1,
            on_step=True, on_epoch=True, prog_bar=True,
        )
        if self.num_classes > 5:
            self.log(
                "val/acc@5", acc5,
                on_step=True, on_epoch=True, prog_bar=True,
            )

    def configure_optimizers(self):
        parameters = set_weight_decay(
            model=self,
            weight_decay=self.weight_decay,
            norm_weight_decay=self.norm_weight_decay,
        )
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})

    return param_groups
