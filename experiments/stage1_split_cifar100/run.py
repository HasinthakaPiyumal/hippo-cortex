"""
Stage 1 — Split-CIFAR100 experiment entry point.

OWNER: Hasinthaka Piyumal

Usage:
    python experiments/stage1_split_cifar100/run.py [key.path=value ...]

    # quick GPU check: 2 tasks, 1 warmup epoch, no W&B
    WANDB_MODE=disabled python experiments/stage1_split_cifar100/run.py \
        dataset.n_tasks=2 training.epochs_warmup=1 training.epochs_joint=0

Reads experiments/stage1_split_cifar100/config.yaml via OmegaConf; command-line
``key.path=value`` arguments override it (an unknown key is an error).
Runs dataset.n_tasks sequential tasks (10 by default); fills acc_matrix (accuracy as
set by eval.mode, class-incremental by default); logs metrics to wandb.

After completing all tasks, saves:
  results/stage1_split_cifar100/acc_matrix.npy
  results/stage1_split_cifar100/summary.json    (AA, AIA, FM, memory_mb)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from hippocortex.utils.seed import set_seed
from hippocortex.utils.config import load_config
from hippocortex.utils.logging import get_logger
from hippocortex.utils import wandb_logger
from hippocortex.utils.metrics import ContinualMetrics
from hippocortex.utils.memory_tracker import MemoryTracker
from hippocortex.models.backbone import MambaBackbone
from hippocortex.models.swr_generator import SWRGenerator
from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.cl.null_space_projector import NullSpaceProjector
from hippocortex.training.trainer import Trainer
from hippocortex.data.split_cifar100 import get_task_loaders, download_if_missing, N_CLASSES

_log = get_logger(__name__)
_CONFIG = Path(__file__).parent / "config.yaml"


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(_CONFIG, overrides=sys.argv[1:] if argv is None else argv)
    n_tasks = int(cfg.dataset.n_tasks)
    per_task = int(cfg.dataset.n_classes_per_task)
    # The split always has 100 / n_classes_per_task tasks; n_tasks may stop early (smoke runs).
    if per_task <= 0 or N_CLASSES % per_task:
        raise ValueError(f"dataset.n_classes_per_task must divide {N_CLASSES}, got {per_task}")
    split_tasks = N_CLASSES // per_task
    if not 1 <= n_tasks <= split_tasks:
        raise ValueError(f"dataset.n_tasks must be in [1, {split_tasks}], got {n_tasks}")
    data_kw = dict(
        seed=int(cfg.seed),
        n_tasks=split_tasks,
        class_order=cfg.dataset.get("class_order", "random"),
        normalize=cfg.dataset.get("normalize", "cifar"),
        num_workers=int(cfg.dataset.get("num_workers", 4)),
        pin_memory=bool(cfg.dataset.get("pin_memory", False)),
    )
    set_seed(cfg.seed)
    wandb_logger.init(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("Using device: %s", device)

    download_if_missing(cfg.dataset.root)

    # ── Model setup ─────────────────────────────────────────────────────────
    bb_cfg = cfg.model.backbone
    backbone = MambaBackbone(
        checkpoint_path=bb_cfg.checkpoint_path,
        config_path=bb_cfg.get("config_path"),
        image_size=bb_cfg.image_size,
        arch=bb_cfg.get("arch", "vim_small"),
        head_mode=bb_cfg.get("head_mode", "per_task"),
        n_classes_total=N_CLASSES,
        grad_ckpt=bool(bb_cfg.get("grad_ckpt", False)),
        hf_dir=bb_cfg.get("hf_dir", "cifar100_10task"),
        seed=int(cfg.seed),
    ).to(device)
    if backbone.feature_dim != cfg.model.mamba_d_model:
        raise ValueError(
            f"model.mamba_d_model={cfg.model.mamba_d_model} but the backbone produces "
            f"{backbone.feature_dim}-d features; the SWR generator and projector use this width."
        )

    swr_gen = SWRGenerator(
        hidden_dim=cfg.model.mamba_d_model,
        latent_dim=cfg.model.cvae_latent_dim,
        n_tasks=cfg.dataset.n_tasks,
    ).to(device)

    buffer = StatsBuffer()
    projector = NullSpaceProjector(rank_budget=cfg.model.nsp_rank_budget)

    trainer = Trainer(backbone, swr_gen, projector, buffer, cfg)

    # ── Training loop ────────────────────────────────────────────────────────
    cl_metrics = ContinualMetrics(num_tasks=n_tasks)
    tracker = MemoryTracker(device=str(device))
    tracker.register_swr_buffer(buffer)
    tracker.start()

    for task_id in range(n_tasks):
        _log.info("── Task %d / %d ──", task_id + 1, n_tasks)

        tracker.task_start(task_id)
        train_loader = get_task_loaders(
            task_id, "train", cfg.dataset.root, cfg.training.batch_size,
            augment=bool(cfg.dataset.get("augment", True)), **data_kw,
        )
        metrics = trainer.train_task(task_id, train_loader)
        tracker.task_end(task_id)

        mem_snap = tracker.task_records[task_id].end_snapshot
        mem_log = {
            "memory/gpu_allocated_mb": mem_snap.gpu_allocated_mb if mem_snap else 0.0,
            "memory/cpu_rss_mb": mem_snap.cpu_rss_mb if mem_snap else 0.0,
            "memory/swr_buffer_mb": mem_snap.swr_buffer_mb if mem_snap else 0.0,
        }

        wandb_logger.log({"task": task_id, **metrics, **mem_log})

        # Evaluate on all tasks seen so far
        accs = {}
        for eval_id in range(task_id + 1):
            test_loader = get_task_loaders(
                eval_id, "test", cfg.dataset.root, int(cfg.eval.get("eval_batch_size", 256)), **data_kw
            )
            acc = trainer.evaluate(eval_id, test_loader)
            cl_metrics.record(task_id, eval_id, acc)
            accs[f"acc/task_{eval_id}"] = acc

        cl_so_far = cl_metrics.summary()
        wandb_logger.log({
            **accs,
            "AA": cl_so_far["final_aa"],
            "FM": cl_so_far["final_fm"],
        })

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
    wandb_logger.log(summary)
    wandb_logger.finish()


if __name__ == "__main__":
    main()
