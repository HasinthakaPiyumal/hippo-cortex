"""
Stage 1 — Split-CIFAR100 experiment entry point.

OWNER: Hasinthaka Piyumal

Usage:
    python experiments/stage1_split_cifar100/run.py

Reads experiments/stage1_split_cifar100/config.yaml via OmegaConf.
Runs 20 sequential tasks; fills acc_matrix; logs metrics to wandb.

After completing all tasks, saves:
  results/stage1_split_cifar100/acc_matrix.npy
  results/stage1_split_cifar100/summary.json    (AA, AF, BWT, memory_mb)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import wandb

from hippocortex.utils.seed import set_seed
from hippocortex.utils.config import load_config
from hippocortex.utils.logging import init_wandb, get_logger
from hippocortex.utils.metrics import build_acc_matrix, average_accuracy, average_forgetting, backward_transfer
from hippocortex.utils.memory_tracker import measure_memory_mb
from hippocortex.models.backbone import MambaBackbone
from hippocortex.models.swr_generator import SWRGenerator
from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.cl.null_space_projector import NullSpaceProjector
from hippocortex.training.trainer import Trainer
from hippocortex.data.split_cifar100 import get_task_loaders, download_if_missing, N_TASKS

_log = get_logger(__name__)
_CONFIG = Path(__file__).parent / "config.yaml"


def main() -> None:
    cfg = load_config(_CONFIG)
    set_seed(cfg.seed)
    init_wandb(cfg, run_name=f"hippocortex_splitcifar100_{cfg.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("Using device: %s", device)

    download_if_missing(cfg.dataset.root)

    # ── Model setup ─────────────────────────────────────────────────────────
    backbone = MambaBackbone(
        d_model=cfg.model.mamba_d_model,
        n_layers=cfg.model.mamba_n_layers,
        d_state=cfg.model.mamba_d_state,
        n_classes=cfg.dataset.n_classes_per_task,
    ).to(device)

    swr_gen = SWRGenerator(
        hidden_dim=cfg.model.mamba_d_model,
        latent_dim=cfg.model.cvae_latent_dim,
        n_tasks=cfg.dataset.n_tasks,
    ).to(device)

    buffer = StatsBuffer()
    projector = NullSpaceProjector(rank_budget=cfg.model.nsp_rank_budget)

    trainer = Trainer(backbone, swr_gen, projector, buffer, cfg)

    # ── Training loop ────────────────────────────────────────────────────────
    acc_matrix = build_acc_matrix(N_TASKS)

    for task_id in range(N_TASKS):
        _log.info("── Task %d / %d ──", task_id + 1, N_TASKS)

        train_loader = get_task_loaders(task_id, "train", cfg.dataset.root, cfg.training.batch_size)
        metrics = trainer.train_task(task_id, train_loader)
        wandb.log({"task": task_id, **metrics})

        # Evaluate on all tasks seen so far
        for eval_id in range(task_id + 1):
            test_loader = get_task_loaders(eval_id, "test", cfg.dataset.root, cfg.training.batch_size)
            acc = trainer.evaluate(eval_id, test_loader)
            acc_matrix[task_id, eval_id] = acc
            wandb.log({f"acc/task_{eval_id}_after_{task_id}": acc})

    # ── Save results ─────────────────────────────────────────────────────────
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "acc_matrix.npy", acc_matrix)

    summary = {
        "AA": average_accuracy(acc_matrix),
        "AF": average_forgetting(acc_matrix),
        "BWT": backward_transfer(acc_matrix),
        "memory_mb": measure_memory_mb(buffer),
        "seed": cfg.seed,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _log.info("Done. AA=%.4f  AF=%.4f  BWT=%.4f", summary["AA"], summary["AF"], summary["BWT"])
    wandb.log(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
