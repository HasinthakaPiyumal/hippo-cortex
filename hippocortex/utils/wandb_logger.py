"""
Thin W&B wrapper for HippoCortex training runs.

All calls are no-ops when cfg.logging.use_wandb is False or wandb is not installed,
so CPU-only runs and unit tests need no W&B account.

Resume support: the W&B run ID is persisted to {checkpoint_dir}/wandb_run_id.txt.
On the next Kaggle session, init() reads that ID and calls wandb.init(resume='allow')
so all tasks land in the same run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_enabled: bool = False
_run = None


def init(cfg: DictConfig) -> None:
    """Initialise a W&B run. No-op if use_wandb is False or wandb is missing."""
    global _enabled, _run

    if not cfg.logging.get("use_wandb", False):
        return

    try:
        import wandb
    except ImportError:
        return

    ckpt_dir = Path(cfg.logging.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_id_file = ckpt_dir / "wandb_run_id.txt"

    run_id = run_id_file.read_text().strip() if run_id_file.exists() else None

    _run = wandb.init(
        project=cfg.logging.get("wandb_project", "hippocortex"),
        entity=cfg.logging.get("wandb_entity") or None,
        name=cfg.logging.get("run_name") or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=run_id,
        resume="allow",
    )
    run_id_file.write_text(_run.id)
    _enabled = True


def log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log a dict of metrics. No-op if disabled."""
    if not _enabled:
        return
    import wandb
    wandb.log(metrics, step=step)


def finish() -> None:
    """Mark the run as finished. No-op if disabled."""
    if not _enabled:
        return
    import wandb
    wandb.finish()
