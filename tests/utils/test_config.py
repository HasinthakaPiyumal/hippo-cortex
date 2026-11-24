"""Tests for load_config command-line overrides (used by experiments/*/run.py)."""
from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from hippocortex.utils.config import load_config

EXPERIMENT_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "stage1_split_cifar100" / "config.yaml"


def test_without_overrides_the_file_is_loaded_as_is():
    cfg = load_config(EXPERIMENT_CONFIG)
    assert cfg.dataset.n_tasks == 10
    assert load_config(EXPERIMENT_CONFIG, overrides=[]) == cfg


def test_overrides_replace_values_and_are_parsed_as_yaml():
    cfg = load_config(
        EXPERIMENT_CONFIG,
        overrides=["dataset.n_tasks=2", "training.epochs_joint=0", "training.grad_clip=null",
                   "training.lr=3e-4", "eval.mode=task_il"],
    )
    assert cfg.dataset.n_tasks == 2
    assert cfg.training.epochs_joint == 0
    assert cfg.training.grad_clip is None
    assert cfg.training.lr == pytest.approx(3e-4)
    assert cfg.eval.mode == "task_il"
    assert cfg.training.epochs_warmup == 5          # untouched keys keep their values


def test_unknown_key_is_an_error():
    with pytest.raises(ConfigKeyError):
        load_config(EXPERIMENT_CONFIG, overrides=["training.epoch_joint=0"])  # typo


def test_loaded_config_stays_writable():
    """Scripts set extra keys after loading (e.g. the smoke script); struct mode is only for the merge."""
    cfg = load_config(EXPERIMENT_CONFIG, overrides=["seed=1"])
    cfg.training.new_key = 1
    assert cfg.training.new_key == 1
