"""
Split-CIFAR100 data loader — 20 sequential tasks of 5 classes each.

OWNER: Induwara Mihisara

Implementation notes:
- Use continuum.ClassIncremental with nb_tasks=20 on torchvision CIFAR-100.
- CIFAR-100 normalization: mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
- The loader must never hold more than one task's data in memory at a time.
- download_if_missing() should be called once at experiment start (run.py).

The task split produced by ClassIncremental must be saved to a JSON file
(data/cifar100/task_split.json) so Thagya's baseline evaluation uses the
identical split.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
N_TASKS = 20
N_CLASSES_PER_TASK = 5


def download_if_missing(root: str | Path) -> None:
    """Download CIFAR-100 to root if not already present."""
    raise NotImplementedError


def get_task_loaders(
    task_id: int,
    split: Literal["train", "val", "test"],
    root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """
    Return a DataLoader for a single task of Split-CIFAR100.

    Args:
        task_id:     Task index in [0, N_TASKS).
        split:       "train", "val", or "test".
        root:        Path to directory where CIFAR-100 is stored/downloaded.
        batch_size:  Samples per mini-batch.
        num_workers: DataLoader worker processes.

    Returns:
        DataLoader yielding (images, labels) where labels are in [0, N_CLASSES_PER_TASK).
    """
    raise NotImplementedError
