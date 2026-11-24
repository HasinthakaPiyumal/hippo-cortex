"""
ImageNet-R data loader: 200 classes split into class-incremental tasks.

OWNER: Hasinthaka Piyumal (was Induwara Mihisara)

Expected layout (what ``scripts/download_datasets.py --dataset imagenet_r`` produces):

    <root>/imagenet-r/<wnid>/<image>.jpg      (200 folders, ~30,000 images)

``root`` may also be the ``imagenet-r`` folder itself.

Two protocols, one per baseline:

- ``protocol="inf_ssm"`` (default; the Vim-small protocol of both baselines, which has released
  task-0 checkpoints for 5 and 10 tasks). Same as ``baselines/inf-ssm/cl_datasets.py``:
    * classes in natural order: task t = classes k*t .. k*t+k-1 (class = index of the sorted
      wnid folder, as torchvision ImageFolder numbers them); 10 tasks by default;
    * one ImageFolder split with ``torch.utils.data.random_split`` and a fresh
      ``torch.Generator`` seeded with 0: 80% train+val / 20% test, then the 80% into
      90% train / 10% val. The same split for every run seed;
    * transforms: Inf-SSM ``build_transform``. Train = timm ``create_transform`` at 224
      (RandAugment, colour jitter, random erasing); eval = resize 256 + centre crop 224.
      ImageNet stats for both.
- ``protocol="mamba_cl"`` (Mamba-CL's own protocol, De-focus backbone):
    * classes shuffled with ``np.random.Generator(np.random.PCG64(seed))`` (seed 2024, as in
      its scripts), then reshaped into tasks; 20 tasks by default
      (``utils/continual_manager.py``);
    * train / val from ``baselines/mamba-cl/tools/imagenet_r_{train,val}.txt``. Mamba-CL has no
      test set; ``split="test"`` returns val;
    * transforms (approximate): train = RandomResizedCrop 224 + flip, eval = resize 256 +
      centre crop 224, ImageNet stats. Mamba-CL itself uses AutoAugment + RandomResizedCrop.

Labels are remapped to [0, classes_per_task) in the order the task's classes are listed, as
in the CIFAR loader; the trainer adds the class offset for a shared head.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from hippocortex.data.split_cifar100 import _inf_ssm_transform

IMAGENET_R_MEAN = (0.485, 0.456, 0.406)
IMAGENET_R_STD = (0.229, 0.224, 0.225)
N_CLASSES = 200
PROTOCOLS = ("inf_ssm", "mamba_cl")
DEFAULT_N_TASKS = {"inf_ssm": 10, "mamba_cl": 20}
DEFAULT_SEED = {"inf_ssm": 0, "mamba_cl": 2024}
# Kept for callers that import the old constants (Mamba-CL protocol).
N_TASKS = 20
N_CLASSES_PER_TASK = 10

_EXTRACTED_DIR = "imagenet-r"
_REPO_ROOT = Path(__file__).resolve().parents[2]
MAMBA_CL_SPLIT_FILES = {
    "train": _REPO_ROOT / "baselines" / "mamba-cl" / "tools" / "imagenet_r_train.txt",
    "val": _REPO_ROOT / "baselines" / "mamba-cl" / "tools" / "imagenet_r_val.txt",
}


def image_dir(root: str | Path) -> Path:
    """The folder holding the 200 wnid folders: <root>/imagenet-r, or root itself."""
    root = Path(root)
    return root / _EXTRACTED_DIR if (root / _EXTRACTED_DIR).is_dir() else root


def is_available(root: str | Path) -> bool:
    d = image_dir(root)
    return d.is_dir() and sum(1 for p in d.iterdir() if p.is_dir()) == N_CLASSES


def _check_protocol(protocol: str) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")


def get_task_classes(
    protocol: str = "inf_ssm", n_tasks: int | None = None, seed: int | None = None
) -> dict[int, list[int]]:
    """Map task_id -> the ImageFolder class indices (sorted wnid order) of that task."""
    _check_protocol(protocol)
    n_tasks = DEFAULT_N_TASKS[protocol] if n_tasks is None else n_tasks
    seed = DEFAULT_SEED[protocol] if seed is None else seed
    if n_tasks <= 0 or N_CLASSES % n_tasks:
        raise ValueError(f"n_tasks must divide {N_CLASSES}, got {n_tasks}")
    k = N_CLASSES // n_tasks
    order = np.arange(N_CLASSES, dtype=np.int64)
    if protocol == "mamba_cl":
        # ClassIncrementalManager(shuffle=True, shuffle_level="class")
        np.random.Generator(np.random.PCG64(seed)).shuffle(order)
    order = order.tolist()
    return {t: order[t * k : (t + 1) * k] for t in range(n_tasks)}


def write_task_split(root: str | Path, protocol: str = "mamba_cl", n_tasks: int | None = None,
                     seed: int | None = None) -> Path:
    """Save the class split as JSON next to the images (e.g. task_split_mamba_cl_seed2024.json)."""
    seed = DEFAULT_SEED[protocol] if seed is None else seed
    classes = get_task_classes(protocol, n_tasks, seed)
    path = Path(root) / f"task_split_{protocol}_seed{seed}.json"
    path.write_text(json.dumps({str(t): c for t, c in classes.items()}, indent=4))
    return path


def build_transform(protocol: str, train: bool, augment: bool = True) -> transforms.Compose:
    _check_protocol(protocol)
    train = train and augment
    if protocol == "inf_ssm":
        return _inf_ssm_transform(train, eval_stats=(IMAGENET_R_MEAN, IMAGENET_R_STD))
    if train:
        steps = [transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.RandomHorizontalFlip()]
    else:
        steps = [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                 transforms.CenterCrop(224)]
    return transforms.Compose(steps + [transforms.ToTensor(),
                                       transforms.Normalize(IMAGENET_R_MEAN, IMAGENET_R_STD)])


@lru_cache(maxsize=None)
def _image_folder(path: str) -> datasets.ImageFolder:
    """The ImageFolder scan without a transform (cached: listing 30k files is slow)."""
    folder = datasets.ImageFolder(path)
    if len(folder.classes) != N_CLASSES:
        raise RuntimeError(f"expected {N_CLASSES} class folders under {path}, found {len(folder.classes)}")
    return folder


@lru_cache(maxsize=None)
def inf_ssm_partition(n_images: int) -> dict[str, list[int]]:
    """Inf-SSM's train / val / test image indices (cl_datasets.build_cl_splits, generator seed 0)."""
    g = torch.Generator()
    g.manual_seed(0)
    n_base = int(0.8 * n_images)
    idx_base, idx_test = torch.utils.data.random_split(range(n_images), [n_base, n_images - n_base], generator=g)
    nb = len(idx_base)
    n_train = int(0.9 * nb)
    idx_train, idx_val = torch.utils.data.random_split(idx_base.indices, [n_train, nb - n_train], generator=g)
    return {
        "train": [idx_base[i] for i in idx_train.indices],
        "val": [idx_base[i] for i in idx_val.indices],
        "test": list(idx_test.indices),
    }


def mamba_cl_partition(folder: datasets.ImageFolder) -> dict[str, list[int]]:
    """Mamba-CL's train / val image indices, from its split files (relative paths 'wnid/file')."""
    index = {Path(p).relative_to(folder.root).as_posix(): i for i, (p, _) in enumerate(folder.samples)}
    out = {}
    for split, path in MAMBA_CL_SPLIT_FILES.items():
        names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        missing = [n for n in names if n not in index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images of {path.name} are not under {folder.root}, "
                                    f"e.g. {missing[0]}")
        out[split] = [index[n] for n in names]
    out["test"] = out["val"]
    return out


class _TaskDataset(Dataset):
    """Images of one task inside one split, labels remapped to [0, classes_per_task)."""

    def __init__(self, folder: datasets.ImageFolder, indices: list[int], classes: list[int], transform):
        self.folder = folder
        self.transform = transform
        self.label_map = {orig: new for new, orig in enumerate(classes)}
        targets = folder.targets
        # Image-folder order, as Inf-SSM's set_task (np.where over the whole folder).
        self.indices = sorted(i for i in indices if targets[i] in self.label_map)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, label = self.folder.samples[self.indices[idx]]
        img = self.folder.loader(path)
        return self.transform(img), self.label_map[label]


def get_task_dataset(
    task_id: int,
    split: Literal["train", "val", "test"],
    root: str | Path,
    protocol: str = "inf_ssm",
    n_tasks: int | None = None,
    seed: int | None = None,
    augment: bool = True,
) -> _TaskDataset:
    _check_protocol(protocol)
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
    task_classes = get_task_classes(protocol, n_tasks, seed)
    if not 0 <= task_id < len(task_classes):
        raise ValueError(f"task_id must be in [0, {len(task_classes)}), got {task_id}")
    d = image_dir(root)
    if not d.is_dir():
        raise FileNotFoundError(f"ImageNet-R not found under {d}. "
                                "Run 'python scripts/download_datasets.py --dataset imagenet_r'.")
    folder = _image_folder(str(d))
    if protocol == "inf_ssm":
        parts = inf_ssm_partition(len(folder.samples))
    else:
        parts = mamba_cl_partition(folder)
    # Inf-SSM builds val with the eval transform (only CIFAR's val uses the train one).
    transform = build_transform(protocol, train=split == "train", augment=augment)
    return _TaskDataset(folder, parts[split], task_classes[task_id], transform)


def get_task_loaders(
    task_id: int,
    split: Literal["train", "val", "test"],
    root: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    protocol: str = "inf_ssm",
    n_tasks: int | None = None,
    seed: int | None = None,
    augment: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Return a DataLoader for a single task of ImageNet-R.

    Args:
        task_id:     Task index in [0, n_tasks).
        split:       "train", "val" or "test" (mamba_cl: "test" is val).
        root:        Directory holding imagenet-r/ (or the imagenet-r folder itself).
        batch_size:  Samples per mini-batch.
        num_workers: DataLoader worker processes.
        protocol:    "inf_ssm" (default) or "mamba_cl"; see the module docstring.
        n_tasks:     Must divide 200 (default 10 for inf_ssm, 20 for mamba_cl).
        seed:        Class-shuffle seed for mamba_cl (default 2024); inf_ssm ignores it
                     (natural order, split generator always seeded with 0).
        augment:     Train augmentation on/off ("train" only).
        pin_memory:  Passed through to DataLoader.

    Returns:
        DataLoader yielding (images (B, 3, 224, 224), labels in [0, 200 / n_tasks)).
        Shuffled for "train" only.
    """
    ds = get_task_dataset(task_id, split, root, protocol, n_tasks, seed, augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train",
                      num_workers=num_workers, pin_memory=pin_memory)
