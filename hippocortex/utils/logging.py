import logging
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
    """Deprecated: use hippocortex.utils.wandb_logger.init(cfg) instead."""
    from hippocortex.utils import wandb_logger
    wandb_logger.init(cfg)
