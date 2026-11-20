"""
Unit tests for hippocortex/utils/wandb_logger.py.

All tests run without a W&B account or network — wandb is fully mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from omegaconf import OmegaConf

# Reset module state between tests
import hippocortex.utils.wandb_logger as wl


def _cfg(use_wandb: bool = True, tmp_ckpt: str = "/tmp/ckpt") -> OmegaConf:
    return OmegaConf.create({
        "logging": {
            "use_wandb": use_wandb,
            "wandb_project": "hippocortex",
            "wandb_entity": None,
            "run_name": "test_run",
            "checkpoint_dir": tmp_ckpt,
        }
    })


def _reset():
    wl._enabled = False
    wl._run = None


# ---------------------------------------------------------------------------
# Disabled (use_wandb: false) — nothing should be called
# ---------------------------------------------------------------------------
class TestDisabled:
    def setup_method(self):
        _reset()

    def test_init_does_not_import_wandb(self):
        with patch.dict(sys.modules, {"wandb": None}):
            wl.init(_cfg(use_wandb=False))
        assert not wl._enabled

    def test_log_is_noop(self):
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.log({"loss": 1.0})
        mock_wandb.log.assert_not_called()

    def test_finish_is_noop(self):
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.finish()
        mock_wandb.finish.assert_not_called()

    def test_missing_wandb_package_is_noop(self, tmp_path):
        cfg = _cfg(use_wandb=True, tmp_ckpt=str(tmp_path))
        with patch.dict(sys.modules, {"wandb": None}):
            wl.init(cfg)
        assert not wl._enabled


# ---------------------------------------------------------------------------
# Enabled — first run (no saved run_id)
# ---------------------------------------------------------------------------
class TestFirstRun:
    def setup_method(self):
        _reset()

    def test_init_calls_wandb_init(self, tmp_path):
        cfg = _cfg(tmp_ckpt=str(tmp_path))
        mock_run = MagicMock()
        mock_run.id = "abc123"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.init(cfg)

        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "hippocortex"
        assert call_kwargs["id"] is None
        assert call_kwargs["resume"] == "allow"

    def test_run_id_saved_to_file(self, tmp_path):
        cfg = _cfg(tmp_ckpt=str(tmp_path))
        mock_run = MagicMock()
        mock_run.id = "new_run_id"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.init(cfg)

        assert (tmp_path / "wandb_run_id.txt").read_text() == "new_run_id"

    def test_enabled_flag_set(self, tmp_path):
        cfg = _cfg(tmp_ckpt=str(tmp_path))
        mock_run = MagicMock()
        mock_run.id = "x"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.init(cfg)

        assert wl._enabled


# ---------------------------------------------------------------------------
# Enabled — resumed run (run_id file already exists)
# ---------------------------------------------------------------------------
class TestResumedRun:
    def setup_method(self):
        _reset()

    def test_init_passes_saved_id(self, tmp_path):
        (tmp_path / "wandb_run_id.txt").write_text("saved_id_xyz")
        cfg = _cfg(tmp_ckpt=str(tmp_path))
        mock_run = MagicMock()
        mock_run.id = "saved_id_xyz"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.init(cfg)

        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["id"] == "saved_id_xyz"
        assert call_kwargs["resume"] == "allow"

    def test_run_id_file_updated_to_returned_id(self, tmp_path):
        (tmp_path / "wandb_run_id.txt").write_text("old_id")
        cfg = _cfg(tmp_ckpt=str(tmp_path))
        mock_run = MagicMock()
        mock_run.id = "returned_id"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.init(cfg)

        assert (tmp_path / "wandb_run_id.txt").read_text() == "returned_id"


# ---------------------------------------------------------------------------
# log() and finish() when enabled
# ---------------------------------------------------------------------------
class TestLogAndFinish:
    def setup_method(self):
        _reset()
        wl._enabled = True  # skip init

    def test_log_calls_wandb_log(self):
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.log({"task": 1, "AA": 28.42, "FM": 5.0})
        mock_wandb.log.assert_called_once_with({"task": 1, "AA": 28.42, "FM": 5.0}, step=None)

    def test_log_passes_step(self):
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.log({"loss": 0.5}, step=42)
        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=42)

    def test_finish_calls_wandb_finish(self):
        mock_wandb = MagicMock()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.finish()
        mock_wandb.finish.assert_called_once()

    def test_log_includes_expected_trainer_keys(self):
        """Verify the keys run.py would log after train_task() match the spec."""
        expected = {"task", "loss_warmup", "loss_joint", "replay_loss",
                    "proj_params_projected", "proj_params_skipped", "loss_consolidation"}
        mock_wandb = MagicMock()
        metrics = {
            "task": 1,
            "loss_warmup": 0.5,
            "loss_joint": 0.0,
            "loss_consolidation": 0.0,
            "replay_loss": 0.0,
            "proj_params_projected": 0,
            "proj_params_skipped": 0,
        }
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            wl.log(metrics)
        logged = set(mock_wandb.log.call_args[0][0].keys())
        assert expected.issubset(logged)
