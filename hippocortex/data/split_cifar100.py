"""
Split-CIFAR100 data loader — 20 tasks of 5 classes by default, or any n_tasks that divides 100.

OWNER: Induwara Mihisara

Expected layout (what ``scripts/download_datasets.py`` produces):

    <root>/cifar-100-python/{meta,train,test}

where ``root`` is ``dataset.root`` in the experiment config (``data/cifar100``).

Implementation notes:
- Class order is ``np.random.default_rng(seed).permutation(100)`` (class_order="random",
  default) or 0..99 (class_order="natural", the Inf-SSM protocol, whose task-0 checkpoint
  was trained on classes 0-9). Task t owns ``order[k*t : k*t+k]`` with k = 100 / n_tasks.
  The default 20-task random split is written to ``<root>/task_split.json`` the first time
  it is built so baselines can reuse it.
- normalize="cifar" (default) uses the CIFAR-100 mean/std; "imagenet" the ImageNet ones,
  which the Vim-small checkpoints expect.
- augment picks the transforms (``AUGMENTS``):
    "cifar" (default, also ``True``): train = random crop 32 + flip; eval = none. Images stay
      32x32 and the backbone upsamples them.
    "none" (also ``False``): no augmentation; eval as "cifar".
    "inf_ssm": exactly ``baselines/inf-ssm/cl_datasets.py::build_transform`` with the
      train_eval.py defaults. Train = timm ``create_transform`` at 224 (RandomResizedCrop,
      flip, RandAugment ``rand-m9-mstd0.5-inc1``, colour jitter 0.3, random erasing 0.25,
      bicubic) normalised with ImageNet stats; eval = resize 256 + centre crop 224 normalised
      with CIFAR-100 stats. ``normalize`` is ignored here: the baseline itself uses ImageNet
      stats for train and CIFAR stats for eval, and we copy that as it is.
  Mixup / cutmix and label smoothing belong to the loss, not the transform, and are not here.
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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_NORMALIZE = {"cifar": (CIFAR100_MEAN, CIFAR100_STD), "imagenet": (IMAGENET_MEAN, IMAGENET_STD)}
CLASS_ORDERS = ("random", "natural")
AUGMENTS = ("cifar", "inf_ssm", "none")

# Inf-SSM transform settings (baselines/inf-ssm/train_eval.py defaults).
INF_SSM_TRANSFORM = dict(
    input_size=224,
    eval_crop_ratio=0.875,
    color_jitter=0.3,
    auto_augment="rand-m9-mstd0.5-inc1",
    interpolation="bicubic",
    re_prob=0.25,
    re_mode="pixel",
    re_count=1,
)
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


def get_task_classes(
    seed: int = 42, n_tasks: int = N_TASKS, class_order: str = "random"
) -> dict[int, list[int]]:
    """Map task_id -> list of the 100 / n_tasks original CIFAR-100 labels of that task."""
    if class_order not in CLASS_ORDERS:
        raise ValueError(f"class_order must be one of {CLASS_ORDERS}, got {class_order!r}")
    if n_tasks <= 0 or N_CLASSES % n_tasks:
        raise ValueError(f"n_tasks must divide {N_CLASSES}, got {n_tasks}")
    k = N_CLASSES // n_tasks
    if class_order == "natural":
        order = list(range(N_CLASSES))
    else:
        order = np.random.default_rng(seed).permutation(N_CLASSES).tolist()
    return {t: order[t * k : (t + 1) * k] for t in range(n_tasks)}


def resolve_augment(augment: bool | str | None) -> str:
    """Map the config value to one of AUGMENTS (True -> "cifar", False / None -> "none")."""
    if augment is True:
        return "cifar"
    if augment is False or augment is None:
        return "none"
    if augment not in AUGMENTS:
        raise ValueError(f"augment must be a bool or one of {AUGMENTS}, got {augment!r}")
    return augment


def _posterize(img, bits_to_keep, **__):
    """timm's posterize with Pillow's old behaviour for 0 bits, as in baselines/inf-ssm/cl_datasets.py.

    RandAugment 'inc1' asks for 4 down to 0 bits; newer Pillow raises on 0 bits inside the
    DataLoader worker, older Pillow returned a black image. Same LUT as ImageOps.posterize.
    """
    if bits_to_keep >= 8:
        return img
    mask = ~(2 ** (8 - bits_to_keep) - 1)
    lut = [i & mask for i in range(256)]
    if img.mode not in ("L", "RGB"):
        raise OSError(f"posterize: mode {img.mode} not supported")
    return img.point(lut * len(img.getbands()))


def _inf_ssm_transform(train: bool) -> transforms.Compose:
    c = INF_SSM_TRANSFORM
    if train:
        import timm.data.auto_augment as timm_aa
        from timm.data import create_transform

        # AugmentOp looks the op up in NAME_TO_OP when it is built, so patching the table is enough.
        for name, fn in list(timm_aa.NAME_TO_OP.items()):
            if fn is timm_aa.posterize:
                timm_aa.NAME_TO_OP[name] = _posterize
        return create_transform(
            input_size=c["input_size"], is_training=True, color_jitter=c["color_jitter"],
            auto_augment=c["auto_augment"], interpolation=c["interpolation"],
            re_prob=c["re_prob"], re_mode=c["re_mode"], re_count=c["re_count"],
        )
    return transforms.Compose([
        transforms.Resize(int(c["input_size"] / c["eval_crop_ratio"]),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(c["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def _build_transform(
    augment: bool | str, normalize: str = "cifar", train: bool = True
) -> transforms.Compose:
    """Transform for one split. ``augment`` only changes the train transform, except "inf_ssm",
    which also has its own eval transform."""
    mode = resolve_augment(augment)
    if mode == "inf_ssm":
        return _inf_ssm_transform(train)
    if normalize not in _NORMALIZE:
        raise ValueError(f"normalize must be one of {tuple(_NORMALIZE)}, got {normalize!r}")
    steps = []
    if mode == "cifar" and train:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize(*_NORMALIZE[normalize])]
    return transforms.Compose(steps)


@lru_cache(maxsize=None)
def _load_cifar100(
    root: str, train: bool, augment: str, normalize: str = "cifar"
) -> datasets.CIFAR100:
    if not is_available(root):
        raise FileNotFoundError(
            f"CIFAR-100 not found under {Path(root) / _EXTRACTED_DIR}. "
            "Run 'python scripts/download_datasets.py --dataset cifar100' "
            "or call download_if_missing(root)."
        )
    return datasets.CIFAR100(
        root=root, train=train, download=False,
        transform=_build_transform(augment, normalize, train=train),
    )


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
    augment: bool | str = True,
    pin_memory: bool = False,
    n_tasks: int = N_TASKS,
    class_order: str = "random",
    normalize: str = "cifar",
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
        augment:     "cifar" / True (crop + flip, train only), "none" / False, or "inf_ssm"
                     (the Inf-SSM train and eval transforms at 224; see the module docstring).
        pin_memory:  Passed through to DataLoader.
        n_tasks:     Number of tasks; must divide 100 (10 x 10 for the Inf-SSM protocol).
        class_order: "random" (seeded permutation) or "natural" (0..99).
        normalize:   "cifar" or "imagenet" mean/std.

    Returns:
        DataLoader yielding (images (B, 3, 32, 32), or (B, 3, 224, 224) with "inf_ssm",
        labels in [0, 100 / n_tasks)).
        Shuffled for "train" only.
    """
    if not (0 <= task_id < n_tasks):
        raise ValueError(f"task_id must be in [0, {n_tasks}), got {task_id}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

    root_path = Path(root)
    train = split == "train"
    mode = resolve_augment(augment)
    # Only "inf_ssm" changes the eval transform; the others share one cached test set.
    if not train and mode != "inf_ssm":
        mode = "none"
    base = _load_cifar100(str(root_path), train, mode, normalize)

    task_classes = get_task_classes(seed, n_tasks, class_order)
    if n_tasks == N_TASKS and class_order == "random":
        _write_task_split(root_path, task_classes, seed)

    return DataLoader(
        _TaskDataset(base, task_classes[task_id]),
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
