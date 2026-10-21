"""Class-incremental datasets, splits and transforms, following vim/datasets.py and the split logic
in vim/main.py of the official Inf-SSM code.

Protocol details kept on purpose, to match the paper's numbers:
- classes are assigned to tasks in their natural order (no shuffling unless --random_class);
- splits use a torch.Generator seeded with 0, so they are the same for every --seed;
- CIFAR-100: val = 10% of the train set and is built with the *train* transform; test = official test set;
- ImageNet-R / Caltech-256: one image folder split 80/20 into train+val / test, then 90/10 into train / val;
- train images are normalised with ImageNet statistics (timm's create_transform default) and eval images
  with dataset statistics (CIFAR-100 stats for CIFAR), exactly as in the official build_transform.
"""
import json
import os

import numpy as np
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder

CIFAR100_DEFAULT_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_DEFAULT_STD = (0.2675, 0.2565, 0.2761)


def partition_dataset(n_classes, n_tasks, randomised=False, seed=0):
    n_classes_per_task = n_classes // n_tasks
    if randomised:
        np.random.seed(seed)
        classes = np.random.permutation(n_classes)
        class_map = {classes[i]: i for i in range(n_classes)}
    else:
        classes = np.array(range(n_classes))
        class_map = {i: i for i in range(n_classes)}
    task_classes = [classes[i * n_classes_per_task: (i + 1) * n_classes_per_task] for i in range(n_tasks)]
    return task_classes, class_map


class _SeqMixin:
    """Restricts a dataset to the samples of one task (set_task) inside one split (set_partition)."""
    num_classes: int
    cur_task_idx = None
    partition_idx = None

    def _init_seq(self, n_tasks, randomised, seed):
        self.n_tasks = n_tasks
        self.n_classes_per_task = self.num_classes // n_tasks
        self.task_classes, self.class_map = partition_dataset(self.num_classes, n_tasks, randomised, seed)
        self.cur_task = 0
        self.cur_task_idx = None

    def partition_transfer(self, train_dataset):
        self.task_classes, self.class_map = train_dataset.task_classes, train_dataset.class_map

    def set_partition(self, partition_idx):
        self.partition_idx = partition_idx

    def set_task(self, task):
        self.cur_task = task
        idx = np.where(np.isin(self.targets, self.task_classes[task]))[0]
        self.cur_task_idx = idx[np.where(np.isin(idx, self.partition_idx))]

    def __len__(self):
        if self.cur_task_idx is not None:
            return len(self.cur_task_idx)
        if self.partition_idx is not None:
            return len(self.partition_idx)
        return super().__len__()

    def __getitem__(self, index):
        img, target = super().__getitem__(self.cur_task_idx[index])
        return img, self.class_map[target]


class SeqCIFAR100(_SeqMixin, datasets.CIFAR100):
    num_classes = 100

    def __init__(self, n_tasks, randomised=False, seed=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_idx = None
        self._init_seq(n_tasks, randomised, seed)


class SeqImageNetR(_SeqMixin, ImageFolder):
    num_classes = 200

    def __init__(self, n_tasks, randomised=False, seed=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_idx = None
        self._init_seq(n_tasks, randomised, seed)


class SeqCaltech256(_SeqMixin, ImageFolder):
    num_classes = 250  # the official code keeps the first 250 of the 257 folders

    def __init__(self, n_tasks, randomised=False, seed=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.partition_idx = np.where(np.isin(self.targets, np.arange(0, self.num_classes, 1)))[0]
        self._init_seq(n_tasks, randomised, seed)


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(input_size=args.input_size, is_training=True, color_jitter=args.color_jitter,
                                     auto_augment=args.aa, interpolation=args.train_interpolation,
                                     re_prob=args.reprob, re_mode=args.remode, re_count=args.recount)
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform
    t = []
    if resize_im:
        t.append(transforms.Resize(int(args.input_size / args.eval_crop_ratio), interpolation=transforms.InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    if args.dataset == 'cifar100':
        t.append(transforms.Normalize(CIFAR100_DEFAULT_MEAN, CIFAR100_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    seq_args = (args.num_tasks, args.random_class, args.seed)
    if args.dataset == 'cifar100':
        dataset = SeqCIFAR100(*seq_args, args.data_root, train=is_train, transform=transform, download=True)
    elif args.dataset == 'imagenet_r':
        dataset = SeqImageNetR(*seq_args, args.data_root, transform=transform)
    elif args.dataset == 'caltech256':
        dataset = SeqCaltech256(*seq_args, args.data_root, transform=transform)
    else:
        raise ValueError(args.dataset)
    return dataset


def build_cl_splits(args, g: torch.Generator, partition_path: str | None = None):
    """Train / val / test datasets with the official splits. `g` must be the fresh seed-0 generator:
    the split draws from it first, then the train DataLoader keeps using it (as in main.py)."""
    if args.dataset in ('imagenet_r', 'caltech256'):
        train, val, test = build_dataset(True, args), build_dataset(False, args), build_dataset(False, args)
        n = len(train)
        idx_base, idx_test = torch.utils.data.random_split(range(n), [int(0.8 * n), n - int(0.8 * n)], generator=g)
        nb = len(idx_base)
        idx_train, idx_val = torch.utils.data.random_split(idx_base.indices, [int(0.9 * nb), nb - int(0.9 * nb)], generator=g)
        if partition_path:
            with open(partition_path, 'w') as f:
                json.dump({'train': idx_train.indices, 'test': idx_test.indices, 'val': idx_val.indices}, f)
        idx_train = [idx_base[i] for i in idx_train.indices]
        idx_val = [idx_base[i] for i in idx_val.indices]
        idx_test = list(idx_test.indices)
        train.set_partition(idx_train)
        val.set_partition(idx_val)
        test.set_partition(idx_test)
    else:
        train, val, test = build_dataset(True, args), build_dataset(True, args), build_dataset(False, args)
        n = len(train)
        idx_train, idx_val = torch.utils.data.random_split(range(n), [int(0.9 * n), n - int(0.9 * n)], generator=g)
        train.set_partition(list(idx_train.indices))
        val.set_partition(list(idx_val.indices))
        test.set_partition(list(range(len(test))))
    val.partition_transfer(train)
    test.partition_transfer(train)
    return train, val, test
