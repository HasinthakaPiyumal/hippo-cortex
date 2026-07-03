"""
Split-CIFAR100 data loader — 20 sequential tasks of 5 classes each.

OWNER: Induwara Mihisara

This implementation uses the robust class split and dataset loading logic
from the original utils/cifar100_dataloader.py wrapper.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
N_TASKS = 20
N_CLASSES_PER_TASK = 5


class SplitCIFAR100Helper:
    def __init__(self, root: str | Path = './data', train: bool = True, seed: int = 42):
        self.root = root
        self.train = train
        self.seed = seed
        self.num_tasks = N_TASKS
        self.classes_per_task = N_CLASSES_PER_TASK

        # Standard CIFAR-100 normalisation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR100_MEAN,
                std=CIFAR100_STD,
            ),
        ])

        # Download/load dataset
        self.dataset = datasets.CIFAR100(
            root=str(self.root),
            train=self.train,
            download=True,
            transform=self.transform,
        )

        # Build class order (fixed seed -> same order every run)
        rng = np.random.default_rng(self.seed)
        self.class_order = rng.permutation(100).tolist()

        # Map each task -> its 5 classes
        self.task_classes = {
            task_id: self.class_order[
                task_id * self.classes_per_task :
                (task_id + 1) * self.classes_per_task
            ]
            for task_id in range(self.num_tasks)
        }

        # Save task split to JSON for baseline consistency
        split_file = Path(self.root) / "cifar100" / "task_split.json"
        if not split_file.exists():
            split_file.parent.mkdir(parents=True, exist_ok=True)
            task_classes_str = {str(k): v for k, v in self.task_classes.items()}
            with open(split_file, "w") as f:
                json.dump(task_classes_str, f, indent=4)

    def get_dataloader(self, task_id: int, batch_size: int = 64, num_workers: int = 4) -> DataLoader:
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(f"task_id must be 0–{self.num_tasks - 1}, got {task_id}")

        classes = self.task_classes[task_id]
        class_set = set(classes)

        # Indices of samples that belong to this task's classes
        indices = [
            i for i, (_, label) in enumerate(self.dataset)
            if label in class_set
        ]

        subset = Subset(self.dataset, indices)

        # Remap original labels -> 0..4
        label_map = {orig: new for new, orig in enumerate(classes)}

        class RemappedSubset(Dataset):
            def __init__(self, subset, label_map):
                self.subset = subset
                self.label_map = label_map

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                img, label = self.subset[idx]
                return img, self.label_map[label]

        remapped = RemappedSubset(subset, label_map)

        return DataLoader(
            remapped,
            batch_size=batch_size,
            shuffle=self.train,
            num_workers=num_workers,
            pin_memory=False,
        )


def download_if_missing(root: str | Path) -> None:
    """Download CIFAR-100 to root if not already present."""
    root_path = Path(root)
    tar_file = root_path / "cifar-100-python.tar.gz"
    extract_dir = root_path / "cifar-100-python"

    if not extract_dir.exists():
        root_path.mkdir(parents=True, exist_ok=True)
        if not tar_file.exists():
            print("Downloading CIFAR-100 from Google Drive via gdown...")
            import gdown
            file_id = "1SjQ7aL1NHX9DqeC72oqmX82lzuv7-FIM"
            gdown.download(id=file_id, output=str(tar_file), quiet=False)

        print("Extracting CIFAR-100 dataset...")
        import tarfile
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=str(root_path))
        print("CIFAR-100 extraction complete.")



def get_task_loaders(
    task_id: int,
    split: Literal["train", "val", "test"],
    root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """
    Return a DataLoader for a single task of Split-CIFAR100.
    """
    train = (split == "train")
    helper = SplitCIFAR100Helper(root=root, train=train, seed=42)
    return helper.get_dataloader(task_id, batch_size=batch_size, num_workers=num_workers)
