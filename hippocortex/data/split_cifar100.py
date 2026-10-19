"""
Split-CIFAR100 data loader — 20 sequential tasks of 5 classes each.

OWNER: Induwara Mihisara

Expected layout (what ``scripts/download_datasets.py`` produces):

    <root>/cifar-100-python/{meta,train,test}

where ``root`` is ``dataset.root`` in the experiment config (``data/cifar100``).

Implementation notes:
- Class order is ``np.random.default_rng(seed).permutation(100)``; task t owns
  classes ``order[5t : 5t+5]``. The split is written to ``<root>/task_split.json``
  the first time it is built so baselines can reuse it.
- Labels are remapped to [0, N_CLASSES_PER_TASK) in the order the classes appear
  in the task's slice of the permutation.
- CIFAR-100 has no validation split; ``split="val"`` returns the test set.
- The underlying torchvision datasets are cached per (root, train, augment), so
  calling ``get_task_loaders`` once per task/eval does not re-read the pickles.
"""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
N_TASKS = 20
N_CLASSES_PER_TASK = 5
N_CLASSES = N_TASKS * N_CLASSES_PER_TASK

_EXTRACTED_DIR = "cifar-100-python"
_DOWNLOAD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_datasets.py"


def is_available(root: str | Path) -> bool:
    """True if the extracted CIFAR-100 python files are under root."""
    extracted = Path(root) / _EXTRACTED_DIR
    return all((extracted / name).exists() for name in ("meta", "train", "test"))


def download_if_missing(root: str | Path) -> None:
    """Download and extract CIFAR-100 into root via scripts/download_datasets.py."""
    if is_available(root):
        return
    spec = importlib.util.spec_from_file_location("_hippocortex_download_datasets", _DOWNLOAD_SCRIPT)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Download script not found at {_DOWNLOAD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.fetch_cifar100(Path(root))


def get_task_classes(seed: int = 42) -> dict[int, list[int]]:
    """Map task_id -> list of the N_CLASSES_PER_TASK original CIFAR-100 labels."""
    order = np.random.default_rng(seed).permutation(N_CLASSES).tolist()
    return {
        t: order[t * N_CLASSES_PER_TASK : (t + 1) * N_CLASSES_PER_TASK]
        for t in range(N_TASKS)
    }


def _build_transform(augment: bool) -> transforms.Compose:
    steps = []
    if augment:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    return transforms.Compose(steps)


@lru_cache(maxsize=None)
def _load_cifar100(root: str, train: bool, augment: bool) -> datasets.CIFAR100:
    if not is_available(root):
        raise FileNotFoundError(
            f"CIFAR-100 not found under {Path(root) / _EXTRACTED_DIR}. "
            "Run 'python scripts/download_datasets.py --dataset cifar100' "
            "or call download_if_missing(root)."
        )
    return datasets.CIFAR100(root=root, train=train, download=False, transform=_build_transform(augment))


def _write_task_split(root: Path, task_classes: dict[int, list[int]], seed: int) -> None:
    split_file = root / "task_split.json" if seed == 42 else root / f"task_split_seed{seed}.json"
    if not split_file.exists():
        split_file.write_text(json.dumps({str(k): v for k, v in task_classes.items()}, indent=4))


class _TaskDataset(Dataset):
    """Samples of one task with labels remapped to [0, N_CLASSES_PER_TASK)."""

    def __init__(self, base: datasets.CIFAR100, classes: list[int]):
        self.base = base
        self.label_map = {orig: new for new, orig in enumerate(classes)}
        # Read labels from .targets rather than iterating the dataset, which
        # would run the image transforms on every sample just to get a label.
        self.indices = [i for i, y in enumerate(base.targets) if y in self.label_map]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.base[self.indices[idx]]
        return img, self.label_map[label]


def get_task_loaders(
    task_id: int,
    split: Literal["train", "val", "test"],
    root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    augment: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Return a DataLoader for a single task of Split-CIFAR100.

    Args:
        task_id:     Task index in [0, N_TASKS).
        split:       "train", "val", or "test" ("val" is the test set).
        root:        Directory containing cifar-100-python/ (e.g. data/cifar100).
        batch_size:  Samples per mini-batch.
        num_workers: DataLoader worker processes (0 is safest on Windows).
        seed:        Seed for the class-to-task permutation.
        augment:     Random crop + horizontal flip; applied to "train" only.
        pin_memory:  Passed through to DataLoader.

    Returns:
        DataLoader yielding (images (B, 3, 32, 32), labels in [0, N_CLASSES_PER_TASK)).
        Shuffled for "train" only.
    """
    if not (0 <= task_id < N_TASKS):
        raise ValueError(f"task_id must be in [0, {N_TASKS}), got {task_id}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

    root_path = Path(root)
    train = split == "train"
    base = _load_cifar100(str(root_path), train, augment and train)

    task_classes = get_task_classes(seed)
    _write_task_split(root_path, task_classes, seed)

    return DataLoader(
        _TaskDataset(base, task_classes[task_id]),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
