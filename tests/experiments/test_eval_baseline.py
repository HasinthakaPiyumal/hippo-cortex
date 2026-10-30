"""
Tests for experiments/stage1_split_cifar100/eval_baseline.py log parser.

Uses a small 3-task fake log at tests/fixtures/mamba_cl_fake_3task.log so
the tests never require a real GPU run or network access.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module under test by path so we don't need a package install.
# ---------------------------------------------------------------------------
_SCRIPT = (
    Path(__file__).parent.parent.parent
    / "experiments" / "stage1_split_cifar100" / "eval_baseline.py"
)

spec = importlib.util.spec_from_file_location("eval_baseline", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

parse_log = _mod.parse_log
extract_final_metrics = _mod.extract_final_metrics
write_csv = _mod.write_csv
plot_acc_curves = _mod.plot_acc_curves

_FIXTURE = Path(__file__).parent.parent / "fixtures" / "mamba_cl_fake_3task.log"


# ---------------------------------------------------------------------------
# parse_log
# ---------------------------------------------------------------------------
class TestParseLog:
    def test_returns_three_records(self):
        records = parse_log(_FIXTURE)
        assert len(records) == 3

    def test_task_ids_are_sequential(self):
        records = parse_log(_FIXTURE)
        assert [r[0] for r in records] == [1, 2, 3]

    def test_first_task_acc(self):
        records = parse_log(_FIXTURE)
        assert records[0][1] == pytest.approx(85.0)

    def test_first_task_forg_zero(self):
        records = parse_log(_FIXTURE)
        assert records[0][2] == pytest.approx(0.0)

    def test_last_task_acc(self):
        records = parse_log(_FIXTURE)
        assert records[-1][1] == pytest.approx(78.30)

    def test_last_task_forg(self):
        records = parse_log(_FIXTURE)
        assert records[-1][2] == pytest.approx(6.20)

    def test_empty_log_returns_empty_list(self, tmp_path):
        empty = tmp_path / "empty.log"
        empty.write_text("no useful lines here\n")
        assert parse_log(empty) == []

    def test_log_with_noise_still_parses(self, tmp_path):
        noisy = tmp_path / "noisy.log"
        noisy.write_text(
            "some irrelevant line\n"
            ":: ** Results of task [1]: [ class_inc_last_acc: 90.00% | class_inc_last_forg:  1.00% ] **\n"
            "more noise\n"
        )
        records = parse_log(noisy)
        assert len(records) == 1
        assert records[0] == (1, 90.0, 1.0)


# ---------------------------------------------------------------------------
# extract_final_metrics
# ---------------------------------------------------------------------------
class TestExtractFinalMetrics:
    def test_returns_last_row_values(self):
        records = parse_log(_FIXTURE)
        aa, fm, n = extract_final_metrics(records)
        assert aa == pytest.approx(78.30)
        assert fm == pytest.approx(6.20)
        assert n == 3

    def test_single_task_log(self):
        aa, fm, n = extract_final_metrics([(1, 85.0, 0.0)])
        assert aa == pytest.approx(85.0)
        assert fm == pytest.approx(0.0)
        assert n == 1

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="No task-result lines"):
            extract_final_metrics([])


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------
class TestWriteCSV:
    def _make_row(self, method="mamba_cl", log_file="run.log"):
        return {
            "method": method,
            "n_tasks": 3,
            "AA": "78.30",
            "FM": "6.20",
            "memory_mb": "",
            "log_file": log_file,
            "eval_date": "2026-10-19",
        }

    def test_creates_csv_with_header(self, tmp_path):
        csv_path = tmp_path / "baselines.csv"
        write_csv([self._make_row()], csv_path)
        assert csv_path.exists()
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            assert "method" in reader.fieldnames
            assert "AA" in reader.fieldnames
            assert "FM" in reader.fieldnames

    def test_writes_correct_values(self, tmp_path):
        csv_path = tmp_path / "baselines.csv"
        write_csv([self._make_row()], csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["method"] == "mamba_cl"
        assert rows[0]["AA"] == "78.30"
        assert rows[0]["FM"] == "6.20"

    def test_overwrites_existing_row_same_method_log(self, tmp_path):
        csv_path = tmp_path / "baselines.csv"
        old_row = self._make_row()
        old_row["AA"] = "50.00"
        write_csv([old_row], csv_path)
        new_row = self._make_row()
        new_row["AA"] = "78.30"
        write_csv([new_row], csv_path)
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
# plot_acc_curves (smoke test only — no visual assertion)
# ---------------------------------------------------------------------------
class TestPlotAccCurves:
    def test_creates_png_file(self, tmp_path):
        records = [(1, 85.0, 0.0), (2, 80.5, 4.5), (3, 78.3, 6.2)]
        plot_path = tmp_path / "acc_curve.png"
        plot_acc_curves([("mamba_cl", records)], plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_multiple_methods_creates_single_file(self, tmp_path):
        r1 = [(1, 85.0, 0.0), (2, 80.5, 4.5)]
        r2 = [(1, 82.0, 0.0), (2, 77.0, 5.0)]
        plot_path = tmp_path / "multi.png"
        plot_acc_curves([("mamba_cl", r1), ("inf_ssm", r2)], plot_path)
        assert plot_path.exists()

    def test_empty_records_does_not_crash(self, tmp_path):
        plot_path = tmp_path / "empty.png"
        plot_acc_curves([("mamba_cl", [])], plot_path)
