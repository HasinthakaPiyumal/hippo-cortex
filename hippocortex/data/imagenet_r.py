"""
ImageNet-R data loader — 20 sequential tasks of 10 classes each.

OWNER: Induwara Mihisara

Implementation notes:
- CRITICAL: The class-to-task assignment MUST match Mamba-CL's split exactly.
  Mamba-CL does NOT use a static JSON file — it generates the split at runtime
  using numpy.random.PCG64(seed) via ClassIncrementalManager (utils/continual_manager.py).
  To reproduce the exact same split: use seed=2024 and n_tasks=20 and replicate
  the shuffle logic:
      rng = np.random.Generator(np.random.PCG64(2024))
      classes = list(range(200))
      rng.shuffle(classes)
      task_class_list = np.array(classes).reshape(20, 10).tolist()
  Save this to data/imagenet_r/task_split_seed2024.json once and load it here.
- ImageNet-R has 200 classes; split into 20 tasks of 10 classes each.
- Normalization: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
- Resize images to 224×224 for the Mamba vision stem.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader


IMAGENET_R_MEAN = (0.485, 0.456, 0.406)
IMAGENET_R_STD = (0.229, 0.224, 0.225)
N_TASKS = 20
N_CLASSES_PER_TASK = 10


def get_task_loaders(
    task_id: int,
    split: Literal["train", "val", "test"],
    root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """
    Return a DataLoader for a single task of ImageNet-R.

    Args:
        task_id:     Task index in [0, N_TASKS).
        split:       "train", "val", or "test".
        root:        Path to directory where ImageNet-R is stored.
        batch_size:  Samples per mini-batch.
        num_workers: DataLoader worker processes.

    Returns:
        DataLoader yielding (images, labels) where labels are in [0, N_CLASSES_PER_TASK).
    """
    raise NotImplementedError
