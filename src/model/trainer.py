"""One training loop shared by ResNet-18 and both EfficientNet models."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .data import build_dataloaders, prepare_group_cache
from .metrics import classification_metrics
from .models import build_model, count_parameters
from .reproducibility import seed_everything


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _state_dict_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    calibration_bins: int,
) -> tuple[float, dict[str, Any]]:
    # eval() disables training behaviors such as dropout. inference_mode()
    # disables gradient tracking because validation/testing never updates weights.
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        total_examples += labels.size(0)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(labels.cpu().numpy())
    metrics = classification_metrics(
        np.concatenate(targets),
        np.concatenate(probabilities),
        class_names,
        calibration_bins,
    )
    return total_loss / total_examples, metrics


def train_epoch(model, loader, criterion, optimizer, scaler, device, mixed_precision: bool) -> float:
    """Perform one complete pass over the training split."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Gradients accumulate by default, so clear the previous batch first.
        optimizer.zero_grad(set_to_none=True)

        # Automatic mixed precision uses smaller GPU number formats where safe.
        # GradScaler prevents very small gradients from becoming zero.
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            logits = model(images)
            loss = criterion(logits, labels)

        # Backpropagation computes gradients; optimizer.step updates the weights.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * labels.size(0)
        total_examples += labels.size(0)
    return total_loss / total_examples


@torch.inference_mode()
def cpu_latency(model: nn.Module, loader, sample_limit: int) -> dict[str, float]:
    # Batch size one approximates a user submitting one image to the future API.
    # CPU timing is recorded because the final hosting service may not have a GPU.
    model = model.to("cpu").eval()
    samples = []
    for batch, _ in loader:
        samples.extend(batch[index : index + 1].cpu() for index in range(batch.size(0)))
        if len(samples) >= sample_limit:
            break
    samples = samples[:sample_limit]
    for image in samples[: min(10, len(samples))]:
        model(image)
    timings = []
    for image in samples:
        start = time.perf_counter()
        model(image)
        timings.append(time.perf_counter() - start)
    return {
        "samples": len(timings),
        "mean_seconds": float(statistics.fmean(timings)),
        "median_seconds": float(statistics.median(timings)),
        "p95_seconds": float(np.percentile(timings, 95)),
    }


def run_group(
    *,
    model_name: str,
    group: str,
    config: dict[str, Any],
    cache_dir: str | Path,
    cache_meta_path: str | Path,
    splits_meta_path: str | Path,
    extract_root: str | Path,
    results_root: str | Path,
    checkpoint_root: str | Path,
) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_root, group_meta = prepare_group_cache(
        cache_dir,
        cache_meta_path,
        splits_meta_path,
        extract_root,
        group,
        expected_seed=seed,
        expected_image_size=int(config["data"]["image_size"]),
        expected_pad_color=list(config["data"]["cache_pad_color"]),
    )
    loaders, class_names, class_weights = build_dataloaders(group_root, group_meta, config)

    # build_model always passes weights=None, so this is random initialization.
    model = build_model(model_name, num_classes=len(class_names)).to(device)
    complexity = count_parameters(model)
    training = config["training"]
    # Cross-entropy compares the five output scores with the correct mineral.
    # Class weights compensate for unequal numbers of training images.
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    # The scheduler gradually reduces the learning rate over the full run.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    mixed_precision = bool(training["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    run_directory = f"seed_{seed}_smoke" if config.get("smoke_test") else f"seed_{seed}"
    result_dir = Path(results_root) / model_name / group / run_directory
    checkpoint_dir = Path(checkpoint_root) / model_name / group / run_directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loaders["train"], criterion, optimizer, scaler, device, mixed_precision)
        val_loss, val_metrics = evaluate(
            model,
            loaders["val"],
            device,
            class_names,
            int(config["evaluation"]["calibration_bins"]),
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        print(record)
        # Select checkpoints using validation data only. Looking at the test set
        # here would leak test information into model selection.
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    training_seconds = time.perf_counter() - started
    # Evaluate the held-out test split once using the best validation checkpoint,
    # rather than whichever epoch happened to finish last.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        class_names,
        int(config["evaluation"]["calibration_bins"]),
    )
    latency = cpu_latency(model, loaders["test"], int(config["evaluation"]["latency_samples"]))

    summary = {
        "protocol_version": config["protocol_version"],
        "model": model_name,
        "group": group,
        "seed": seed,
        "classes": class_names,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "training_seconds": training_seconds,
        "assimilation_time_seconds": training_seconds,
        "cpu_inference_latency": latency,
        "complexity": complexity,
        "checkpoint_sha256": _state_dict_sha256(checkpoint_path),
        "git_commit": _git_commit(),
        "device": str(device),
        "hardware": {
            "processor": platform.processor() or "unknown",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
        "shared_config": config,
    }
    _json_write(result_dir / "history.json", history)
    _json_write(result_dir / "summary.json", summary)
    return summary

