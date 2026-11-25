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
compute_bwt = _mod.compute_bwt
write_csv = _mod.write_csv
write_main_table = _mod.write_main_table
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

    def test_mamba_cl_shared_head_run_logs_inf_ssm_keys(self):
        # Mamba-CL trained with the Inf-SSM code (shared head) logs AA/AIA/FM.
        rows = [
            {"task": 0, "epoch": 39, "train/loss": 0.3},
            {"task": 0, "AA": 0.97, "AIA": 0.97, "FM": 0.0},
            {"task": 9, "AA": 0.5152, "AIA": 0.5943, "FM": 0.4862},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/mamba-cl-baseline/3388e092e0b1", "mamba_cl")
        assert result["AA"] == pytest.approx(51.52)
        assert result["AIA"] == pytest.approx(59.43)
        assert result["FM"] == pytest.approx(48.62)
        assert result["n_tasks"] == 2

    def test_scan_history_has_no_keys_filter(self):
        # A keys= filter makes W&B drop rows that lack any listed key.
        run = _make_wandb_run([{"task": 1, "AA": 50.0, "FM": 5.0}])
        with self._patch_api(run):
            fetch_wandb_run("entity/p/abc", "mamba_cl")
        _, kwargs = run.scan_history.call_args
        assert "keys" not in kwargs

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

    # --- TIL mapping ---
    def test_til_mapped_and_scaled(self):
        rows = [
            {"task": 0, "AA": 0.90, "AIA": 0.90, "FM": 0.0, "TIL_AA": 0.95},
            {"task": 1, "AA": 0.825, "AIA": 0.8625, "FM": 0.05, "TIL_AA": 0.92},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")
        assert result["TIL"] == pytest.approx(92.0)

    def test_til_none_when_not_logged(self):
        rows = [{"task": 1, "class_inc_last_acc": 85.0, "class_inc_last_forg": 0.0}]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/mamba-cl-baseline/abc", "mamba_cl")
        assert result["TIL"] is None

    # --- BWT from the inf-ssm-protocol test/class_<i>_acc matrix ---
    def test_bwt_reconstructed_from_class_acc_columns(self):
        # Matches the hand-computed example in TestComputeBwt.
        rows = [
            {"task": 0, "AA": 0.90, "AIA": 0.90, "FM": 0.0, "test/class_0_acc": 0.90},
            {"task": 1, "AA": 0.825, "AIA": 0.8625, "FM": 0.05,
             "test/class_0_acc": 0.85, "test/class_1_acc": 0.80},
            {"task": 2, "AA": 0.8267, "AIA": 0.85, "FM": 0.05,
             "test/class_0_acc": 0.82, "test/class_1_acc": 0.78, "test/class_2_acc": 0.88},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")
        assert result["BWT"] == pytest.approx(-5.0)

    def test_bwt_none_when_matrix_not_logged(self):
        # Mamba-CL's own protocol never logs test/class_<i>_acc.
        rows = [
            {"task": 1, "class_inc_last_acc": 85.0, "class_inc_last_forg": 0.0},
            {"task": 2, "class_inc_last_acc": 90.16, "class_inc_last_forg": 2.18},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/mamba-cl-baseline/abc", "mamba_cl")
        assert result["BWT"] is None

    def test_bwt_none_with_single_task(self):
        rows = [{"task": 0, "AA": 0.90, "AIA": 0.90, "FM": 0.0, "test/class_0_acc": 0.90}]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/inf-ssm-baseline/abc", "inf_ssm")
        assert result["BWT"] is None

    # --- hippocortex key mapping ---
    def test_hippocortex_key_mapping(self):
        rows = [
            {"task": 0, "AA": 0.90, "AIA": 0.90, "FM": 0.0},
            {"task": 1, "AA": 0.825, "AIA": 0.8625, "FM": 0.05},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/hippocortex-stage1/abc", "hippocortex")
        assert result["AA"] == pytest.approx(82.5)
        assert result["AIA"] == pytest.approx(86.25)
        assert result["FM"] == pytest.approx(5.0)
        assert result["n_tasks"] == 2

    # --- BWT from HippoCortex's run.py acc/task_<i> matrix ---
    def test_bwt_reconstructed_from_acc_task_columns(self):
        # Same hand-computed example as TestComputeBwt.test_three_task_example,
        # but using HippoCortex's acc/task_<i> column naming instead of
        # Inf-SSM's test/class_<i>_acc.
        rows = [
            {"task": 0, "AA": 0.90, "AIA": 0.90, "FM": 0.0, "acc/task_0": 0.90},
            {"task": 1, "AA": 0.825, "AIA": 0.8625, "FM": 0.05,
             "acc/task_0": 0.85, "acc/task_1": 0.80},
            {"task": 2, "AA": 0.8267, "AIA": 0.85, "FM": 0.05,
             "acc/task_0": 0.82, "acc/task_1": 0.78, "acc/task_2": 0.88},
        ]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/hippocortex-stage1/abc", "hippocortex")
        assert result["BWT"] == pytest.approx(-5.0)

    def test_hippocortex_bwt_none_with_single_task(self):
        rows = [{"task": 0, "AA": 0.90, "AIA": 0.90, "FM": 0.0, "acc/task_0": 0.90}]
        with self._patch_api(_make_wandb_run(rows)):
            result = fetch_wandb_run("entity/hippocortex-stage1/abc", "hippocortex")
        assert result["BWT"] is None


# ---------------------------------------------------------------------------
# compute_bwt — pure function, hand-checkable
# ---------------------------------------------------------------------------
class TestComputeBwt:
    def test_three_task_example(self):
        # M[0] = [0.90]
        # M[1] = [0.85, 0.80]
        # M[2] = [0.82, 0.78, 0.88]
        # BWT = mean_{i=0,1}(M[2][i] - M[i][i])
        #     = ((0.82-0.90) + (0.78-0.80)) / 2 = -0.05 -> -5.0 pp
        class_accs = {
            0: {0: 0.90},
            1: {0: 0.85, 1: 0.80},
            2: {0: 0.82, 1: 0.78, 2: 0.88},
        }
        assert compute_bwt(class_accs) == pytest.approx(-5.0)

    def test_positive_backward_transfer(self):
        # Task 0 improves after later training: BWT > 0.
        class_accs = {
            0: {0: 0.70},
            1: {0: 0.80, 1: 0.75},
        }
        assert compute_bwt(class_accs) == pytest.approx(10.0)

    def test_single_task_returns_none(self):
        assert compute_bwt({0: {0: 0.90}}) is None

    def test_empty_returns_none(self):
        assert compute_bwt({}) is None

    def test_missing_diagonal_entry_is_skipped(self):
        # Task 1's own diagonal (class_1_acc at t=1) was never logged.
        class_accs = {
            0: {0: 0.90},
            1: {0: 0.85},  # no key 1 here
            2: {0: 0.82, 1: 0.78, 2: 0.88},
        }
        # Only task 0 contributes: (0.82 - 0.90) = -0.08 -> -8.0 pp
        assert compute_bwt(class_accs) == pytest.approx(-8.0)


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


# ---------------------------------------------------------------------------
# write_main_table
# ---------------------------------------------------------------------------
class TestWriteMainTable:
    def _rows(self):
        return [
            {"method": "inf_ssm", "AA": "28.42", "AIA": "55.35", "FM": "24.83",
             "TIL": "70.99", "BWT": "-5.00",
             "wandb_run": "hasinthakapiyumal-uok/inf-ssm-baseline/d591d88c21e6",
             "eval_date": "2026-11-22"},
            {"method": "mamba_cl_shared_head", "AA": "51.52", "AIA": "59.43", "FM": "48.62",
             "TIL": "94.94", "BWT": "",
             "wandb_run": "hasinthakapiyumal-uok/mamba-cl-baseline/3388e092e0b1",
             "eval_date": "2026-11-22"},
        ]

    def test_writes_csv_with_all_columns(self, tmp_path):
        write_main_table(self._rows(), tmp_path)
        with open(tmp_path / "main_table.csv", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for key in ("method", "AA", "AIA", "FM", "TIL", "BWT", "wandb_run", "eval_date"):
                assert key in reader.fieldnames
        assert len(rows) == 2
        assert rows[0]["AA"] == "28.42"

    def test_writes_markdown_table(self, tmp_path):
        write_main_table(self._rows(), tmp_path)
        text = (tmp_path / "main_table.md").read_text()
        assert "inf_ssm" in text
        assert "28.42" in text
        assert "mamba_cl_shared_head" in text
        assert "d591d88c21e6" in text

    def test_markdown_includes_protocol_footnote(self, tmp_path):
        write_main_table(self._rows(), tmp_path)
        text = (tmp_path / "main_table.md").read_text()
        assert "null-space" in text.lower()

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        write_main_table(self._rows(), nested)
        assert (nested / "main_table.csv").exists()
        assert (nested / "main_table.md").exists()
