"""
Stage 1 — Split-CIFAR100 experiment entry point.

OWNER: Hasinthaka Piyumal

Usage:
    python experiments/stage1_split_cifar100/run.py

Reads experiments/stage1_split_cifar100/config.yaml via OmegaConf.
Runs 20 sequential tasks; fills acc_matrix; logs metrics to wandb.

After completing all tasks, saves:
  results/stage1_split_cifar100/acc_matrix.npy
  results/stage1_split_cifar100/summary.json    (AA, AIA, FM, memory_mb)
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
from hippocortex.utils.metrics import ContinualMetrics
from hippocortex.utils.memory_tracker import MemoryTracker
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
    cl_metrics = ContinualMetrics(num_tasks=N_TASKS)
    tracker = MemoryTracker(device=str(device))
    tracker.register_swr_buffer(buffer)
    tracker.start()

    for task_id in range(N_TASKS):
        _log.info("── Task %d / %d ──", task_id + 1, N_TASKS)

        tracker.task_start(task_id)
        train_loader = get_task_loaders(task_id, "train", cfg.dataset.root, cfg.training.batch_size)
        metrics = trainer.train_task(task_id, train_loader)
        tracker.task_end(task_id)
        wandb.log({"task": task_id, **metrics})

        # Evaluate on all tasks seen so far
        for eval_id in range(task_id + 1):
            test_loader = get_task_loaders(eval_id, "test", cfg.dataset.root, cfg.training.batch_size)
            acc = trainer.evaluate(eval_id, test_loader)
            cl_metrics.record(task_id, eval_id, acc)
            wandb.log({f"acc/task_{eval_id}_after_{task_id}": acc})

    tracker.stop()

    # ── Save results ─────────────────────────────────────────────────────────
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "acc_matrix.npy", cl_metrics.result_matrix.to_tensor().numpy())

    cl_summary = cl_metrics.summary()
    mem_summary = tracker.summary()
    summary = {
        "AA": cl_summary["final_aa"],
        "AIA": cl_summary["final_aia"],
        "FM": cl_summary["final_fm"],
        "memory_mb": buffer.memory_bytes() / (1024 ** 2),
        "peak_gpu_mb": mem_summary["peak_gpu_mb"],
        "seed": cfg.seed,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _log.info("Done. AA=%.4f  AIA=%.4f  FM=%.4f", summary["AA"], summary["AIA"], summary["FM"])
    wandb.log(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
