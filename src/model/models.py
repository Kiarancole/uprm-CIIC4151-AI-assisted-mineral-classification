"""One model factory prevents architecture-specific training notebooks."""

from __future__ import annotations

from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b3, resnet18


def build_model(name: str, num_classes: int = 5) -> nn.Module:
    """Construct a model from random initialization.

    `weights=None` is intentional: the proposal specifies training from scratch.
    The same seed controls reproducibility, but different architectures cannot
    have identical tensors because their layer shapes differ.
    """
    builders = {
        "resnet18": resnet18,
        "efficientnet_b0": efficientnet_b0,
        "efficientnet_b3": efficientnet_b3,
    }
    try:
        builder = builders[name]
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(builders)}") from error
    return builder(weights=None, num_classes=num_classes)


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }

