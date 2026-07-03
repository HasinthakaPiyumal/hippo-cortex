import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


class SplitCIFAR100:
    """
    Splits CIFAR-100 into 20 tasks with 5 classes each.
    Matches Mamba-CL's split for directly comparable results.
    """

    def __init__(self, root='./data', train=True, batch_size=32, seed=42):
        """
        Args:
            root       : where to download CIFAR-100
            train      : True = training set, False = test set
            batch_size : how many images per batch
            seed       : fixed seed so split is always the same
        """
        self.root       = root
        self.train      = train
        self.batch_size = batch_size
        self.seed       = seed
        self.num_tasks  = 20
        self.classes_per_task = 5   # 20 tasks x 5 classes = 100 classes

        # ── Standard CIFAR-100 normalisation ──────────────────────────────
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std =(0.2675, 0.2565, 0.2761),
            ),
        ])

        # ── Download dataset ───────────────────────────────────────────────
        self.dataset = datasets.CIFAR100(
            root=self.root,
            train=self.train,
            download=True,
            transform=self.transform,
        )

        # ── Build class order (fixed seed → same order every run) ──────────
        rng = np.random.default_rng(self.seed)
        self.class_order = rng.permutation(100).tolist()

        # ── Map each task → its 5 classes ─────────────────────────────────
        self.task_classes = {
            task_id: self.class_order[
                task_id * self.classes_per_task :
                (task_id + 1) * self.classes_per_task
            ]
            for task_id in range(self.num_tasks)
        }

    # ──────────────────────────────────────────────────────────────────────
    def get_dataloader(self, task_id: int) -> DataLoader:
        """
        Returns a DataLoader for the given task (0-indexed, 0–19).
        Labels are remapped to 0–4 within each task.
        """
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(f"task_id must be 0–{self.num_tasks - 1}, got {task_id}")

        classes   = self.task_classes[task_id]
        class_set = set(classes)

        # Indices of samples that belong to this task's classes
        indices = [
            i for i, (_, label) in enumerate(self.dataset)
            if label in class_set
        ]

        subset = Subset(self.dataset, indices)

        # Remap original labels → 0..4 so the model always sees 0-based labels
        label_map = {orig: new for new, orig in enumerate(classes)}

        class RemappedSubset(torch.utils.data.Dataset):
            def __init__(self, subset, label_map):
                self.subset    = subset
                self.label_map = label_map

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                img, label = self.subset[idx]
                return img, self.label_map[label]

        remapped = RemappedSubset(subset, label_map)

        return DataLoader(
            remapped,
            batch_size=self.batch_size,
            shuffle=self.train,   # shuffle only for training
            num_workers=0,        # 0 = safe on Windows
            pin_memory=False,
        )

    # ──────────────────────────────────────────────────────────────────────
    def get_all_dataloaders(self):
        """Returns a list of DataLoaders for all 20 tasks."""
        return [self.get_dataloader(task_id) for task_id in range(self.num_tasks)]


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading SplitCIFAR100 ...")
    split = SplitCIFAR100(root='E:/data', train=True, batch_size=32)

    for task_id in range(20):
        loader  = split.get_dataloader(task_id)
        classes = split.task_classes[task_id]
        imgs, labels = next(iter(loader))
        print(f"Task {task_id + 1:02d} | classes: {classes} | "
              f"batch shape: {imgs.shape} | labels: {labels.tolist()[:5]}")

    print("\nAll 20 tasks loaded successfully!")