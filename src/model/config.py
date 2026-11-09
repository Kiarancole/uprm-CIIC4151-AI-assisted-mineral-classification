"""Load and validate the frozen preliminary-experiment configuration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


SUPPORTED_MODELS = {"resnet18", "efficientnet_b0", "efficientnet_b3"}
SUPPORTED_GROUPS = {"high", "medium", "low"}


def load_config(path: str | Path) -> dict[str, Any]:
    # Keeping experiment choices in one committed file prevents teammates from
    # silently training with different learning rates, epochs, or augmentations.
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    # Fail early if a setting no longer matches the agreed preliminary protocol.
    # This is safer than discovering an unfair comparison after hours of training.
    required = {
        "protocol_version",
        "seed",
        "groups",
        "models",
        "data",
        "training",
        "evaluation",
        "priority_weights",
        "selection",
        "success_thresholds",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Configuration is missing fields: {sorted(missing)}")

    if set(config["models"]) != SUPPORTED_MODELS:
        raise ValueError(f"models must contain exactly {sorted(SUPPORTED_MODELS)}")
    if set(config["groups"]) != SUPPORTED_GROUPS:
        raise ValueError(f"groups must contain exactly {sorted(SUPPORTED_GROUPS)}")
    if config["data"]["image_size"] != 224:
        raise ValueError("The shared cache and preliminary protocol require 224x224 images")
    if list(config["data"]["cache_pad_color"]) != [124, 116, 104]:
        raise ValueError("cache_pad_color must match the merged 224x224 cache metadata")
    if config["training"]["optimizer"].lower() != "adamw":
        raise ValueError("The preliminary protocol currently supports AdamW only")
    if config["training"]["loss"] != "balanced_cross_entropy":
        raise ValueError("The preliminary protocol requires balanced_cross_entropy")
    if int(config["training"]["epochs"]) <= 0 or int(config["data"]["batch_size"]) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if int(config["evaluation"]["calibration_bins"]) != 10:
        raise ValueError("The proposal defines ECE using exactly 10 confidence bins")
    if set(config["priority_weights"]) != SUPPORTED_GROUPS:
        raise ValueError("priority_weights must contain high, medium, and low")

    priority_total = sum(float(value) for value in config["priority_weights"].values())
    if abs(priority_total - 1.0) > 1e-9:
        raise ValueError("priority_weights must add up to 1.0")


def smoke_test_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy for pipeline verification, never for official results."""
    updated = deepcopy(config)
    updated["training"]["epochs"] = 1
    updated["smoke_test"] = True
    return updated

