"""
Baseline evaluation script — produces results/stage1_split_cifar100/baselines.csv.

OWNER: Thagya Kavindi

Usage:
    python experiments/stage1_split_cifar100/eval_baseline.py \
        --mamba_cl_ckpt /tmp/mamba-cl-ref/checkpoints/

This script evaluates Mamba-CL checkpoints using HippoCortex's own data loaders
so the comparison in Paper 1 Table 1 is apples-to-apples.

Output columns:
    method, task_id, accuracy, AA, AF, BWT, memory_mb, ckpt_commit, eval_date
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from hippocortex.utils.config import load_config
from hippocortex.utils.metrics import build_acc_matrix, average_accuracy, average_forgetting, backward_transfer
from hippocortex.data.split_cifar100 import get_task_loaders, N_TASKS

_CONFIG = Path(__file__).parent / "config.yaml"
_SAVE_DIR = Path("results/stage1_split_cifar100")
_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_mamba_cl(mamba_cl_ckpt_dir: Path) -> dict:
    """Load Mamba-CL checkpoints and evaluate on HippoCortex data loaders."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba_cl_ckpt", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(_CONFIG)
    results = []

    print("Evaluating Mamba-CL baseline ...")
    mamba_cl_results = evaluate_mamba_cl(args.mamba_cl_ckpt)
    results.append({"method": "mamba_cl", **mamba_cl_results, "eval_date": str(date.today())})

    # Write CSV
    csv_path = _SAVE_DIR / "baselines.csv"
    fieldnames = ["method", "AA", "AF", "BWT", "memory_mb", "ckpt_commit", "eval_date"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
