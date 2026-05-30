"""
experiments/stage1_split_cifar100/run_finetune.py
==================================================
Sequential fine-tuning baseline for Split-CIFAR100.

This is the **lower-bound** ("Seq" / "naive fine-tuning") baseline used
throughout the continual learning literature and reported as "Seq" in the
Inf-SSM paper (Lee et al. 2025, Tables 4 & 5).

The model trains on Task 1, then Task 2, … Task T with **no** replay,
**no** regularisation, and **no** gradient projection.  Each new task's
data simply overwrites prior knowledge — catastrophic forgetting in its
purest form.  Every other method in Stage 1 must beat these numbers.

Usage
-----
    python experiments/stage1_split_cifar100/run_finetune.py

    # Override defaults via CLI:
    python experiments/stage1_split_cifar100/run_finetune.py \
        --n-tasks 20 --epochs 5 --batch-size 64 --lr 1e-3 \
        --seed 42 --device cuda --results-dir results/finetune

Integration with Hasinthaka's trainer
--------------------------------------
When the shared trainer module is ready, replace the local
`_train_one_epoch` and `_evaluate` stubs with:

    from hippocortex.trainer import Trainer
    trainer = Trainer(model, ...)
    trainer.fit(task_train_loader)
    acc = trainer.evaluate(task_test_loader)

Everything else (the outer task loop, the R-matrix accumulation, and
the final metric computation) stays unchanged.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
from utils.metrics import average_accuracy, average_incremental_accuracy, forgetting_measure


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
N_CLASSES     = 100   # total CIFAR-100 classes


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def get_cifar100(data_root: str = "./data") -> tuple:
    """Download CIFAR-100 and return (train_dataset, test_dataset)."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_ds = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform_test
    )
    return train_ds, test_ds


def make_task_splits(
    train_ds, test_ds, n_tasks: int, seed: int
) -> List[tuple]:
    """
    Split 100 CIFAR-100 classes into n_tasks disjoint groups.

    Returns a list of (train_subset, test_subset, class_indices) tuples,
    one per task.  Class indices are shuffled deterministically by `seed`
    so results are reproducible.
    """
    rng = random.Random(seed)
    classes = list(range(N_CLASSES))
    rng.shuffle(classes)

    classes_per_task = N_CLASSES // n_tasks
    task_splits = []
    for t in range(n_tasks):
        task_classes = set(classes[t * classes_per_task: (t + 1) * classes_per_task])

        train_idx = [i for i, (_, y) in enumerate(train_ds) if y in task_classes]
        test_idx  = [i for i, (_, y) in enumerate(test_ds)  if y in task_classes]

        task_splits.append((
            Subset(train_ds, train_idx),
            Subset(test_ds,  test_idx),
            sorted(task_classes),
        ))
    return task_splits


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_classes: int, device: torch.device) -> nn.Module:
    """
    Placeholder backbone.

    Once Hasinthaka's Mamba backbone is integrated, replace this with:
        from hippocortex.backbone import MambaBackbone
        model = MambaBackbone(n_classes=n_classes)

    For now we use a small ResNet-18 so the pipeline can be tested
    end-to-end immediately.
    """
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


# ---------------------------------------------------------------------------
# Training & evaluation stubs
# (replace internals with Hasinthaka's trainer when ready)
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch.  Returns average cross-entropy loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return top-1 accuracy in [0, 100]."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    """
    Full sequential fine-tuning run.

    Returns
    -------
    dict with keys: AA, AIA, FM, R
    """
    # ---- Reproducibility ----
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[run_finetune] device={device}  n_tasks={args.n_tasks}  "
          f"epochs={args.epochs}  seed={args.seed}")

    # ---- Data ----
    print("[run_finetune] Loading CIFAR-100 …")
    train_ds, test_ds = get_cifar100(args.data_root)
    task_splits = make_task_splits(train_ds, test_ds, args.n_tasks, args.seed)
    n_classes_per_task = N_CLASSES // args.n_tasks

    # ---- Model ----
    # NOTE: we build a single shared-head model.  In a task-incremental
    # setup you might also use per-task heads; adjust here when the
    # backbone interface is finalised.
    model = build_model(N_CLASSES, device)
    criterion = nn.CrossEntropyLoss()

    # ---- Result matrix R (0-based) ----
    # R[i][j] = accuracy on task j after training through task i
    T = args.n_tasks
    R: List[List[float]] = [[0.0] * T for _ in range(T)]

    # Build all test loaders upfront (we'll evaluate on every seen task
    # after each training task).
    test_loaders = [
        DataLoader(
            task_splits[t][1],
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for t in range(T)
    ]

    # ---- Outer task loop ----
    for task_idx in range(T):
        train_subset, _, task_classes = task_splits[task_idx]

        print(f"\n{'='*60}")
        print(f"[Task {task_idx + 1}/{T}]  classes={task_classes}")
        print(f"{'='*60}")

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        # Fresh optimizer each task — the defining feature of naive fine-tuning.
        # (No EWC penalty, no gradient projection, no replay buffer.)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        # ---- Inner epoch loop ----
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}  ({elapsed:.1f}s)")

        # ---- Evaluate on every task seen so far ----
        print(f"\n  [Evaluation after task {task_idx + 1}]")
        for j in range(task_idx + 1):
            acc = _evaluate(model, test_loaders[j], device)
            R[task_idx][j] = acc
            print(f"    task {j + 1:3d} accuracy: {acc:.2f}%")

    # ---- Final metrics ----
    aa  = average_accuracy(R)
    aia = average_incremental_accuracy(R)
    fm  = forgetting_measure(R)

    print(f"\n{'='*60}")
    print("FINAL METRICS  (Seq / naive fine-tuning lower bound)")
    print(f"{'='*60}")
    print(f"  Average Accuracy       (AA)  = {aa:.2f}%")
    print(f"  Avg Incremental Acc   (AIA)  = {aia:.2f}%")
    print(f"  Forgetting Measure     (FM)  = {fm:.2f}%")

    # ---- Save results ----
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    import json
    out = {
        "method":  "finetune",
        "dataset": "split_cifar100",
        "n_tasks": T,
        "epochs":  args.epochs,
        "seed":    args.seed,
        "AA":      aa,
        "AIA":     aia,
        "FM":      fm,
        "R":       R,
    }
    out_path = results_dir / f"finetune_T{T}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential fine-tuning baseline on Split-CIFAR100."
    )
    p.add_argument("--n-tasks",     type=int,   default=20,
                   help="Number of tasks to split CIFAR-100 into (default 20)")
    p.add_argument("--epochs",      type=int,   default=5,
                   help="Training epochs per task (default 5)")
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      type=str,   default="cuda",
                   help="'cuda' or 'cpu'")
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--data-root",   type=str,   default="./data")
    p.add_argument("--results-dir", type=str,   default="results/finetune")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)