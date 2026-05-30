"""
experiments/stage1_split_cifar100/run_finetune.py
──────────────────────────────────────────────────
Sequential Fine-Tuning Baseline (Lower Bound)

This is the **simplest possible continual learner**: train on each task in
sequence, no replay buffer, no projector, no regularisation.  Every gradient
update overwrites whatever the network learned on previous tasks.  The result
is the *worst-case* catastrophic-forgetting curve that every other method in
the paper must beat.

In the milestone doc this is labelled "Naive Fine-Tuning (No CL)" and shown
as the dashed line in Figure 5.2.  In the Inf-SSM paper (Tab. 1/4/5) the
equivalent entry is "Seq".

Usage
─────
    python experiments/stage1_split_cifar100/run_finetune.py [--options]

    # minimal run (CPU, quick smoke-test with fewer epochs):
    python experiments/stage1_split_cifar100/run_finetune.py \\
        --epochs 5 --device cpu

    # full run matching Inf-SSM paper protocol:
    python experiments/stage1_split_cifar100/run_finetune.py \\
        --epochs 50 --device cuda --num_tasks 10 --seed 42

Outputs
───────
    results/finetune/R_matrix.npy   — T×T accuracy matrix
    results/finetune/metrics.json   — {AA, AIA, FM}
    results/finetune/run_log.txt    — per-epoch loss / accuracy

Plug-in point for Hasinthaka's trainer
───────────────────────────────────────
    Replace the `train_one_task()` function below with a call to
    hippocortex.trainer.Trainer.  The rest of the script (data loading,
    evaluation loop, metric computation) stays exactly the same.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ── project imports ───────────────────────────────────────────────────
# Make sure PYTHONPATH includes the repo root, or install the package:
#   pip install -e .
from hippocortex.utils.metrics import compute_all   # AA, AIA, FM

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential fine-tuning lower bound on Split-CIFAR-100"
    )
    p.add_argument("--data_root",  default="~/data", type=str,
                   help="Root directory for CIFAR-100 download/cache.")
    p.add_argument("--out_dir",    default="results/finetune", type=str,
                   help="Directory to save R matrix, metrics, and log.")
    p.add_argument("--num_tasks",  default=10, type=int,
                   help="Number of equal-sized class-incremental tasks.")
    p.add_argument("--epochs",     default=50, type=int,
                   help="Training epochs per task.")
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--lr",         default=1e-3, type=float)
    p.add_argument("--device",     default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--seed",       default=42, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
    ])
    return train_tf, test_tf


def make_task_splits(
    root: str,
    num_tasks: int,
    train_tf,
    test_tf,
) -> tuple[list[DataLoader], list[DataLoader]]:
    """
    Split CIFAR-100 into `num_tasks` class-incremental tasks.

    CIFAR-100 has 100 classes; with num_tasks=10 each task gets 10 classes.
    Classes are sorted and split sequentially (same as Inf-SSM / Mamba-CL
    benchmark protocol).

    Returns
    -------
    train_loaders, test_loaders : lists of length num_tasks
    """
    root = os.path.expanduser(root)
    full_train = datasets.CIFAR100(root, train=True,  download=True, transform=train_tf)
    full_test  = datasets.CIFAR100(root, train=False, download=True, transform=test_tf)

    total_classes = 100
    classes_per_task = total_classes // num_tasks
    assert total_classes % num_tasks == 0, (
        f"100 classes must be divisible by num_tasks; got {num_tasks}"
    )

    train_loaders, test_loaders = [], []

    for t in range(num_tasks):
        class_start = t * classes_per_task
        class_end   = class_start + classes_per_task
        task_classes = list(range(class_start, class_end))

        # Indices of samples that belong to this task's classes
        tr_idx = [i for i, (_, y) in enumerate(full_train)
                  if y in task_classes]
        te_idx = [i for i, (_, y) in enumerate(full_test)
                  if y in task_classes]

        train_loaders.append(
            DataLoader(Subset(full_train, tr_idx), batch_size=64,
                       shuffle=True, num_workers=4, pin_memory=True)
        )
        test_loaders.append(
            DataLoader(Subset(full_test, te_idx), batch_size=256,
                       shuffle=False, num_workers=4, pin_memory=True)
        )

    return train_loaders, test_loaders


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 100) -> nn.Module:
    """
    ResNet-18 with a fresh linear head for *all* 100 CIFAR-100 classes.

    Using a shared head (all 100 outputs always visible) is the standard
    "single-head" evaluation protocol used by Inf-SSM and most exemplar-free
    CL papers.  Task ID is NOT given at inference time.

    Swap this function out if the team decides on a different backbone
    (e.g., Mamba-CL's SSM encoder) — nothing else needs to change.
    """
    model = models.resnet18(weights=None)
    # Adapt for 32×32 CIFAR images (remove the aggressive downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────

def train_one_task(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    task_idx: int,
) -> None:
    """
    Fine-tune `model` on a single task for `epochs` epochs.

    ┌─────────────────────────────────────────────────────────┐
    │  PLUG-IN POINT FOR HASINTHAKA'S TRAINER                 │
    │                                                         │
    │  Replace the body of this function with:                │
    │                                                         │
    │      from hippocortex.trainer import Trainer            │
    │      trainer = Trainer(model, cfg)                      │
    │      trainer.fit(loader, task_id=task_idx)              │
    │                                                         │
    │  The R-matrix evaluation loop below stays unchanged.    │
    └─────────────────────────────────────────────────────────┘

    This baseline version: plain cross-entropy + Adam, no regularisation,
    no replay, no projector — pure catastrophic forgetting.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += x.size(0)

        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            log.info(
                "  Task %d | epoch %3d/%d | loss %.4f | train acc %.2f%%",
                task_idx + 1, epoch, epochs,
                total_loss / total, 100.0 * correct / total,
            )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return accuracy (%) on the given data loader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred   = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += x.size(0)
    return 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device != "cuda"
        else "cpu"
    )

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("HippoCortex — Sequential Fine-Tuning Baseline (lower bound)")
    log.info("  num_tasks=%d | epochs/task=%d | device=%s | seed=%d",
             args.num_tasks, args.epochs, device, args.seed)
    log.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────
    train_tf, test_tf = build_transforms()
    train_loaders, test_loaders = make_task_splits(
        args.data_root, args.num_tasks, train_tf, test_tf
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(num_classes=100).to(device)
    log.info("Backbone: ResNet-18 (CIFAR variant), %.2fM params",
             sum(p.numel() for p in model.parameters()) / 1e6)

    # ── R matrix: R[i][j] = acc on task j after training on task i ───
    T = args.num_tasks
    R = np.zeros((T, T), dtype=np.float64)

    t0 = time.time()

    for task_i in range(T):
        log.info("─" * 50)
        log.info(">>> Training on task %d / %d", task_i + 1, T)

        # Fine-tune on task i (no replay, no regularisation)
        train_one_task(
            model, train_loaders[task_i],
            epochs=args.epochs, lr=args.lr,
            device=device, task_idx=task_i,
        )

        # Evaluate on ALL tasks seen so far (fill row i of R)
        log.info("    Evaluating on tasks 1 – %d …", task_i + 1)
        for task_j in range(task_i + 1):
            acc = evaluate(model, test_loaders[task_j], device)
            R[task_i, task_j] = acc
            log.info("      a_{%d,%d} = %.2f%%", task_i + 1, task_j + 1, acc)

    elapsed = time.time() - t0
    log.info("Total training time: %.1f s (%.1f min)", elapsed, elapsed / 60)

    # ── Compute metrics ───────────────────────────────────────────────
    metrics = compute_all(R)
    log.info("=" * 60)
    log.info("RESULTS (Sequential Fine-Tuning — Lower Bound)")
    log.info("  AA  = %.2f%%  ↑ (higher is better)", metrics["AA"])
    log.info("  AIA = %.2f%%  ↑ (higher is better)", metrics["AIA"])
    log.info("  FM  = %.2f%%  ↓ (lower is better)",  metrics["FM"])
    log.info("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────
    np.save(out_dir / "R_matrix.npy", R)
    log.info("Saved R matrix → %s", out_dir / "R_matrix.npy")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "method": "sequential_finetune",
                "num_tasks": T,
                "epochs_per_task": args.epochs,
                "seed": args.seed,
                "AA":  metrics["AA"],
                "AIA": metrics["AIA"],
                "FM":  metrics["FM"],
            },
            f, indent=2,
        )
    log.info("Saved metrics   → %s", out_dir / "metrics.json")

    # Pretty-print the R matrix for the log
    log.info("\nR matrix (R[i][j] = acc on task j+1 after task i+1):\n%s",
             np.array2string(R, formatter={"float_kind": lambda x: f"{x:6.2f}"}))


if __name__ == "__main__":
    main()