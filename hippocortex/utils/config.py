from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """
    Load a YAML config file into an OmegaConf DictConfig.

    Args:
        path:      YAML file.
        overrides: Command-line style ``key.path=value`` strings, e.g.
                   ``["dataset.n_tasks=2", "training.grad_clip=null"]``. Values are
                   parsed as YAML. A key that is not in the file raises, so a typo
                   cannot silently do nothing.
    """
    cfg = OmegaConf.load(str(path))
    if overrides:
        OmegaConf.set_struct(cfg, True)  # reject keys the file does not define
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        OmegaConf.set_struct(cfg, False)
    return cfg
