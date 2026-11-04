"""Verify, extract, and load the shared 224x224 group caches."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .reproducibility import seed_worker, seeded_generator


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _safe_extract(zip_path: Path, destination: Path) -> None:
    # Reject archive entries that try to write outside the chosen cache folder.
    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise ValueError(f"Unsafe path in cache archive: {member.filename}")
        archive.extractall(destination)


def prepare_group_cache(
    cache_dir: str | Path,
    cache_meta_path: str | Path,
    splits_meta_path: str | Path,
    extract_root: str | Path,
    group: str,
    expected_seed: int,
    expected_image_size: int,
    expected_pad_color: list[int],
) -> tuple[Path, dict[str, Any]]:
    metadata = load_cache_metadata(cache_meta_path)
    splits_metadata = load_cache_metadata(splits_meta_path)
    # Confirm that the merged cache notebook produced data compatible with this
    # training protocol before spending time hashing or training.
    if int(splits_metadata["seed"]) != expected_seed:
        raise ValueError(
            f"Dataset split seed is {splits_metadata['seed']}, but the protocol expects "
            f"{expected_seed}"
        )
    if metadata["splits_csv_sha256"] != splits_metadata["splits_csv_sha256"]:
        raise ValueError("Cache metadata was created from a different splits.csv file")
    if int(metadata["img_size"]) != expected_image_size:
        raise ValueError(
            f"Cache image size is {metadata['img_size']}, but the protocol expects "
            f"{expected_image_size}"
        )
    if list(metadata["pad_color"]) != list(expected_pad_color):
        raise ValueError(
            f"Cache pad color is {metadata['pad_color']}, but the protocol expects "
            f"{expected_pad_color}"
        )
    if group not in metadata["groups"]:
        raise ValueError(f"Unknown group {group!r}; choose from {sorted(metadata['groups'])}")

    group_meta = metadata["groups"][group]
    if set(group_meta["labels"]) != set(splits_metadata["groups"][group]):
        raise ValueError(f"Cache and split metadata disagree about the minerals in {group}")
    zip_path = Path(cache_dir) / group_meta["zip"]
    if not zip_path.is_file():
        raise FileNotFoundError(f"Cache archive not found: {zip_path}")

    # A checksum proves that everyone is using the exact same cached images.
    actual_hash = sha256_file(zip_path)
    expected_hash = group_meta["zip_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"Checksum mismatch for {zip_path.name}: expected {expected_hash}, found {actual_hash}")

    # Colab trains faster when the archive is copied/extracted to its temporary
    # local disk instead of reading every image directly from Google Drive.
    group_root = Path(extract_root) / group
    complete_marker = group_root / f".complete-{expected_hash}"
    if not complete_marker.exists():
        if group_root.exists():
            shutil.rmtree(group_root)
        group_root.mkdir(parents=True, exist_ok=True)
        _safe_extract(zip_path, group_root)
        complete_marker.touch()

    return group_root, group_meta


def build_transforms(config: dict[str, Any]) -> tuple[transforms.Compose, transforms.Compose]:
    data_config = config["data"]
    augmentation = data_config["augmentation"]
    # Normalization rescales color channels to a stable numeric range. Using the
    # common ImageNet mean/std does NOT load ImageNet weights; models still use
    # random initialization because models.py passes weights=None.
    normalize = transforms.Normalize(
        mean=data_config["normalization_mean"],
        std=data_config["normalization_std"],
    )
    # Random augmentation is training-only. Validation and test images must stay
    # unchanged so every model is measured on the same inputs.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(augmentation["horizontal_flip_probability"]),
            transforms.RandomVerticalFlip(augmentation["vertical_flip_probability"]),
            transforms.RandomRotation(
                augmentation["rotation_degrees"],
                interpolation=InterpolationMode.BILINEAR,
                fill=tuple(augmentation["rotation_fill"]),
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, evaluation_transform


def build_dataloaders(
    group_root: Path,
    group_meta: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, DataLoader], list[str], torch.Tensor]:
    train_transform, evaluation_transform = build_transforms(config)
    # ImageFolder infers labels from the directory layout:
    # <group>/<split>/<mineral-name>/<image>.jpg
    datasets_by_split = {
        "train": datasets.ImageFolder(group_root / "train", transform=train_transform),
        "val": datasets.ImageFolder(group_root / "val", transform=evaluation_transform),
        "test": datasets.ImageFolder(group_root / "test", transform=evaluation_transform),
    }

    # Verify label numbers rather than assuming alphabetical folders always match
    # the metadata produced by the shared cache notebook.
    expected_labels = {name: int(label) for name, label in group_meta["labels"].items()}
    for split, dataset in datasets_by_split.items():
        if dataset.class_to_idx != expected_labels:
            raise ValueError(
                f"{split} class mapping differs from cache metadata: "
                f"expected {expected_labels}, found {dataset.class_to_idx}"
            )
        expected_count = int(group_meta["counts"][split])
        if len(dataset) != expected_count:
            raise ValueError(
                f"{split} contains {len(dataset)} images, but cache metadata expects "
                f"{expected_count}"
            )

    seed = int(config["seed"])
    batch_size = int(config["data"]["batch_size"])
    workers = int(config["data"]["num_workers"])
    # Only the training loader shuffles. The seeded generator and worker setup
    # make the batch order and random augmentations reproducible.
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=seeded_generator(seed),
            persistent_workers=workers > 0,
        )
        for split, dataset in datasets_by_split.items()
    }

    targets = np.asarray(datasets_by_split["train"].targets)
    counts = np.bincount(targets, minlength=len(expected_labels))
    if np.any(counts == 0):
        raise ValueError(f"Training split has an empty class: {counts.tolist()}")
    # Inverse-frequency weights give minerals with fewer training images more
    # influence in the loss instead of letting the largest class dominate.
    class_weights = len(targets) / (len(counts) * counts)
    class_names = [name for name, _ in sorted(expected_labels.items(), key=lambda item: item[1])]
    return loaders, class_names, torch.tensor(class_weights, dtype=torch.float32)

