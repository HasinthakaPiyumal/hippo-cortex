from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file into an OmegaConf DictConfig."""
    return OmegaConf.load(str(path))
