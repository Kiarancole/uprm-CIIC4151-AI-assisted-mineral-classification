"""Utilities that keep model initialization and data loading reproducible."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    # Model initialization, augmentations, and batch shuffling each use random
    # numbers. Seeding every library makes repeated runs as comparable as the
    # available PyTorch operations and hardware allow.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN benchmarking may choose a different algorithm between executions.
    # Deterministic mode prioritizes repeatability over maximum training speed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    # Every DataLoader worker is a separate process and therefore needs a seed.
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

