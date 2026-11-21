"""
Tests for experiments/stage1_split_cifar100/eval_baseline.py.

Uses:
  - tests/fixtures/mamba_cl_fake_3task.log  — log-parser tests
  - unittest.mock to fake wandb.Api          — W&B reader tests (no network, no key)
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by path (no package install needed)
# ---------------------------------------------------------------------------
_SCRIPT = (
    Path(__file__).parent.parent.parent
    / "experiments" / "stage1_split_cifar100" / "eval_baseline.py"
)
spec = importlib.util.spec_from_file_location("eval_baseline", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)
sys.modules["eval_baseline"] = _mod  # needed so patch("eval_baseline.wandb.Api") resolves
spec.loader.exec_module(_mod)

parse_log = _mod.parse_log
extract_final_metrics_from_log = _mod.extract_final_metrics_from_log
fetch_wandb_run = _mod.fetch_wandb_run
write_csv = _mod.write_csv
plot_acc_curves = _mod.plot_acc_curves

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "mamba_cl_fake_3task.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_wandb_run(history_rows: list[dict]) -> MagicMock:
    """Return a fake wandb Run whose scan_history yields the given rows."""
    run = MagicMock()
    run.scan_history.return_value = iter(history_rows)
    return run


# ---------------------------------------------------------------------------
# parse_log (log fallback)
# ---------------------------------------------------------------------------
class TestParseLog:
    def test_returns_three_records(self):
        assert len(parse_log(_FIXTURE)) == 3

    def test_task_ids_are_sequential(self):
        assert [r[0] for r in parse_log(_FIXTURE)] == [1, 2, 3]

    def test_first_task_acc(self):
        assert parse_log(_FIXTURE)[0][1] == pytest.approx(85.0)

    def test_first_task_forg_zero(self):
        assert parse_log(_FIXTURE)[0][2] == pytest.approx(0.0)

    def test_last_task_acc(self):
        assert parse_log(_FIXTURE)[-1][1] == pytest.approx(78.30)

    def test_last_task_forg(self):
        assert parse_log(_FIXTURE)[-1][2] == pytest.approx(6.20)

    def test_empty_log_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.log"
        f.write_text("no useful lines here\n")
        assert parse_log(f) == []

    def test_log_with_noise_still_parses(self, tmp_path):
        f = tmp_path / "noisy.log"
        f.write_text(
            "some irrelevant line\n"
            ":: ** Results of task [1]: [ class_inc_last_acc: 90.00% | class_inc_last_forg:  1.00% ] **\n"
            "more noise\n"
        )
        records = parse_log(f)
        assert len(records) == 1
        assert records[0] == (1, 90.0, 1.0)


# ---------------------------------------------------------------------------
# extract_final_metrics_from_log
# ---------------------------------------------------------------------------
class TestExtractFinalMetricsFromLog:
    def test_returns_last_row_values(self):
        result = extract_final_metrics_from_log(parse_log(_FIXTURE))
        assert result["AA"] == pytest.approx(78.30)
        assert result["FM"] == pytest.approx(6.20)
        assert result["n_tasks"] == 3

    def test_aia_is_none_for_log_source(self):
        result = extract_final_metrics_from_log(parse_log(_FIXTURE))
        assert result["AIA"] is None

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="No task-result lines"):
            extract_final_metrics_from_log([])


# ---------------------------------------------------------------------------
# fetch_wandb_run — fully mocked, no network or API key needed
# ---------------------------------------------------------------------------
class TestFetchWandbRun:
    def _patch_api(self, run):
        """Context manager that replaces wandb.Api().run() with `run`.

        wandb is imported inside fetch_wandb_run so we patch the real
        wandb.Api, not a module-level attribute.
        """
        mock_api_instance = MagicMock()
        mock_api_instance.run.return_value = run
        return patch("wandb.Api", return_value=mock_api_instance)

    # --- inf_ssm key mapping ---
    def test_inf_ssm_basic(self):
        rows = [
            {"task": 1, "AA": 60.0, "AIA": 70.0, "FM": 5.0},
            {"task": 2, "AA": 55.0, "AIA": 65.0, "FM": 8.0},
            {"task": 3, "AA": 28.42, "AIA": 55.35, "FM": 24.83},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")
        assert result["AA"] == pytest.approx(28.42)
        assert result["AIA"] == pytest.approx(55.35)
        assert result["FM"] == pytest.approx(24.83)
        assert result["n_tasks"] == 3

    def test_inf_ssm_records_order(self):
        rows = [
            {"task": 2, "AA": 55.0, "AIA": 65.0, "FM": 8.0},
            {"task": 1, "AA": 60.0, "AIA": 70.0, "FM": 5.0},
            {"task": 3, "AA": 28.42, "AIA": 55.35, "FM": 24.83},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")
        assert [r[0] for r in result["records"]] == [1, 2, 3]

    # --- mamba_cl key mapping ---
    def test_mamba_cl_key_mapping(self):
        rows = [
            {"task": 1, "class_inc_last_acc": 85.0, "class_inc_last_forg": 0.0},
            {"task": 2, "class_inc_last_acc": 80.5, "class_inc_last_forg": 4.5},
            {"task": 3, "class_inc_last_acc": 90.16, "class_inc_last_forg": 2.18},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/mamba-cl-baseline/abc", "mamba_cl")
        assert result["AA"] == pytest.approx(90.16)
        assert result["FM"] == pytest.approx(2.18)
        assert result["AIA"] is None  # not logged by mamba_cl

    # --- duplicate task (resumed session) — last row wins ---
    def test_duplicate_task_keeps_last(self):
        rows = [
            {"task": 1, "AA": 60.0, "AIA": 70.0, "FM": 5.0},
            {"task": 2, "AA": 50.0, "AIA": 60.0, "FM": 10.0},  # first attempt
            {"task": 2, "AA": 55.0, "AIA": 65.0, "FM": 8.0},   # resumed, supersedes
            {"task": 3, "AA": 28.42, "AIA": 55.35, "FM": 24.83},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")
        task2_row = next(r for r in result["records"] if r[0] == 2)
        assert task2_row[1] == pytest.approx(55.0)  # last value kept

    def test_duplicate_task_does_not_inflate_n_tasks(self):
        rows = [
            {"task": 1, "AA": 60.0, "AIA": 70.0, "FM": 5.0},
            {"task": 1, "AA": 62.0, "AIA": 72.0, "FM": 4.0},  # repeated
            {"task": 2, "AA": 55.0, "AIA": 65.0, "FM": 8.0},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")
        assert result["n_tasks"] == 2

    # --- rows without task key are ignored ---
    def test_non_task_rows_ignored(self):
        rows = [
            {"train/loss": 1.2},                                # epoch row — no task key
            {"task": 1, "AA": 60.0, "AIA": 70.0, "FM": 5.0},
            {"train/loss": 0.8},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")
        assert result["n_tasks"] == 1

    # --- empty run raises ---
    def test_empty_run_raises(self):
        with self._patch_api(_make_wandb_run([])):
            with pytest.raises(ValueError, match="No per-task rows"):
                fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------
class TestWriteCSV:
    def _make_row(self, method="mamba_cl", source="run.log"):
        return {
            "method": method,
            "n_tasks": 3,
            "AA": "78.30",
            "AIA": "",
            "FM": "6.20",
            "source": source,
            "eval_date": "2026-10-30",
        }

    def test_creates_csv_with_header(self, tmp_path):
        csv_path = tmp_path / "baselines.csv"
        write_csv([self._make_row()], csv_path)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for key in ("method", "AA", "FM", "AIA", "source"):
                assert key in reader.fieldnames

    def test_writes_correct_values(self, tmp_path):
        csv_path = tmp_path / "baselines.csv"
        write_csv([self._make_row()], csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["AA"] == "78.30"
        assert rows[0]["FM"] == "6.20"

    def test_overwrites_existing_row_same_method_source(self, tmp_path):
        csv_path = tmp_path / "baselines.csv"
        old = self._make_row(); old["AA"] = "50.00"
        write_csv([old], csv_path)
        new = self._make_row(); new["AA"] = "78.30"
        write_csv([new], csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["AA"] == "78.30"

    def test_keeps_different_methods(self, tmp_path):
        csv_path = tmp_path / "baselines.csv"
        write_csv([self._make_row("mamba_cl", "a.log")], csv_path)
        write_csv([self._make_row("inf_ssm", "b.log")], csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir" / "baselines.csv"
        write_csv([self._make_row()], nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# plot_acc_curves
# ---------------------------------------------------------------------------
class TestPlotAccCurves:
    def test_creates_png_file(self, tmp_path):
        records = [(1, 85.0, 0.0), (2, 80.5, 4.5), (3, 78.3, 6.2)]
        plot_path = tmp_path / "acc_curve.png"
        plot_acc_curves([("mamba_cl", records)], plot_path)
        assert plot_path.exists() and plot_path.stat().st_size > 0

    def test_multiple_methods(self, tmp_path):
        plot_path = tmp_path / "multi.png"
        plot_acc_curves(
            [("mamba_cl", [(1, 85.0, 0.0), (2, 80.5, 4.5)]),
             ("inf_ssm",  [(1, 82.0, 0.0), (2, 77.0, 5.0)])],
            plot_path,
        )
        assert plot_path.exists()

    def test_empty_records_does_not_crash(self, tmp_path):
        plot_acc_curves([("mamba_cl", [])], tmp_path / "empty.png")
