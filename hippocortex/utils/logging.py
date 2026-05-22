import logging
import wandb
from omegaconf import DictConfig, OmegaConf


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def init_wandb(cfg: DictConfig, run_name: str | None = None) -> None:
    """Initialise a wandb run from config. No-op if wandb is disabled."""
    wandb.init(
        project=cfg.logging.wandb_project,
        name=run_name or cfg.logging.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
