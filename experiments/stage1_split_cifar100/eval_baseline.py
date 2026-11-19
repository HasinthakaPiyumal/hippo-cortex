"""
Baseline evaluation script — produces results/stage1_split_cifar100/baselines.csv.

OWNER: Thagya Kavindi

Primary source: W&B (reads per-task rows from a finished run via the public API).
Fallback source: log file (regex parser for Mamba-CL task-result lines).

Usage — W&B:
    python experiments/stage1_split_cifar100/eval_baseline.py \\
        --wandb_run hasinthakapiyumal-uok/inf-ssm-baseline/d591d88c21e6 --method inf_ssm

Usage — log fallback:
    python experiments/stage1_split_cifar100/eval_baseline.py \\
        --log path/to/run.log --method mamba_cl

Multiple methods in one call (mix sources freely):
    python experiments/stage1_split_cifar100/eval_baseline.py \\
        --wandb_run hasinthakapiyumal-uok/mamba-cl-baseline/abc123 --method mamba_cl \\
        --log logs/inf_ssm.log --method inf_ssm

W&B API key:
    Set WANDB_API_KEY env var (local login or Kaggle secret). Never commit it.

Output CSV columns:
    method, n_tasks, AA, AIA, FM, source, eval_date

NOTE: BWT requires the per-task accuracy matrix, which is not logged by default.
Tell Hasinthaka to add a matrix-print line to train_eval.py if BWT is needed.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SAVE_DIR = Path("results/stage1_split_cifar100")

# ---------------------------------------------------------------------------
# Column name mapping per W&B project → common names (AA, AIA, FM)
# ---------------------------------------------------------------------------
# Each entry: wandb_key -> common_key
_WANDB_KEY_MAP: Dict[str, Dict[str, str]] = {
    "mamba_cl": {
        "task": "task",
        "class_inc_last_acc": "AA",
        "class_inc_last_forg": "FM",
    },
    "inf_ssm": {
        "task": "task",
        "AA": "AA",
        "AIA": "AIA",
        "FM": "FM",
    },
}

# Fallback for unknown methods — try these common key names
_WANDB_FALLBACK_KEYS = {
    "task": "task",
    "AA": "AA",
    "AIA": "AIA",
    "FM": "FM",
    "class_inc_last_acc": "AA",
    "class_inc_last_forg": "FM",
}

# ---------------------------------------------------------------------------
# Log-file parser (kept as fallback)
# ---------------------------------------------------------------------------
_TASK_LINE_RE = re.compile(
    r"Results of task \[(\d+)\].*?"
    r"class_inc_last_acc:\s*([\d.]+)%.*?"
    r"class_inc_last_forg:\s*([\d.]+)%"
)


def parse_log(log_path: Path) -> List[Tuple[int, float, float]]:
    """
    Return [(task_id, acc_pct, forg_pct), ...] in log order (1-indexed task_id).
    acc and forg are percentages, e.g. 91.20 not 0.9120.
    """
    records: List[Tuple[int, float, float]] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in _TASK_LINE_RE.finditer(text):
        records.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return records


def extract_final_metrics_from_log(
    records: List[Tuple[int, float, float]],
) -> Dict[str, object]:
    """Return {AA, FM, n_tasks} from the last log entry."""
    if not records:
        raise ValueError("No task-result lines found in the log.")
    task_id, aa_pct, fm_pct = records[-1]
    return {"AA": aa_pct, "AIA": None, "FM": fm_pct, "n_tasks": task_id, "records": records}


# ---------------------------------------------------------------------------
# W&B reader
# ---------------------------------------------------------------------------
def fetch_wandb_run(run_path: str, method: str) -> Dict[str, object]:
    """
    Pull per-task rows from a W&B run and return:
        {"AA": float, "AIA": float|None, "FM": float, "n_tasks": int,
         "records": [(task_id, AA_pct, FM_pct), ...]}

    run_path: "entity/project/run_id"
    Reads WANDB_API_KEY from the environment (wandb login also works).
    If a task appears more than once (resumed session), keeps the last row.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is not installed. Run: pip install wandb")

    api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))
    run = api.run(run_path)

    key_map = _WANDB_KEY_MAP.get(method, _WANDB_FALLBACK_KEYS)
    needed_wandb_keys = list(key_map.keys())

    # scan_history returns all rows; filter to those that have the task key
    task_key = "task"
    rows_by_task: Dict[int, Dict[str, float]] = {}

    for row in run.scan_history(keys=needed_wandb_keys):
        if task_key not in row or row[task_key] is None:
            continue
        try:
            task_id = int(row[task_key])
        except (TypeError, ValueError):
            continue
        mapped = {}
        for wk, ck in key_map.items():
            if wk in row and row[wk] is not None:
                v = float(row[wk])
                # W&B may log fractions (0–1); convert to percentage
                if ck in ("AA", "AIA", "FM") and v <= 1.0:
                    v *= 100.0
                mapped[ck] = v
        if "AA" in mapped:
            rows_by_task[task_id] = mapped  # last row for this task wins

    if not rows_by_task:
        raise ValueError(
            f"No per-task rows with key 'AA' found in W&B run {run_path}.\n"
            f"Keys seen: {needed_wandb_keys}. "
            "Check that the run has finished at least one task."
        )

    sorted_tasks = sorted(rows_by_task.keys())
    records = [(t, rows_by_task[t]["AA"], rows_by_task[t].get("FM", float("nan")))
               for t in sorted_tasks]

    last = rows_by_task[sorted_tasks[-1]]
    return {
        "AA": last["AA"],
        "AIA": last.get("AIA"),
        "FM": last.get("FM"),
        "n_tasks": len(sorted_tasks),
        "records": records,
    }


# ---------------------------------------------------------------------------
# CSV + plot
# ---------------------------------------------------------------------------
def write_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "n_tasks", "AA", "AIA", "FM", "source", "eval_date"]
    existing: List[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    existing = [
        r for r in existing
        if not any(
            r.get("method") == row["method"] and r.get("source") == row["source"]
            for row in rows
        )
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing + rows)


def plot_acc_curves(
    method_records: List[Tuple[str, List[Tuple[int, float, float]]]],
    plot_path: Path,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, records in method_records:
        if not records:
            continue
        tasks = [r[0] for r in records]
        accs = [r[1] for r in records]
        ax.plot(tasks, accs, marker="o", label=method)
    ax.set_xlabel("Task")
    ax.set_ylabel("Class-incremental Accuracy (%)")
    ax.set_title("Accuracy vs Task (Split CIFAR-100)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch baseline results from W&B (or log fallback) → CSV + plot."
    )
    parser.add_argument(
        "--wandb_run", action="append", dest="wandb_runs", metavar="ENTITY/PROJECT/RUN_ID",
        help="W&B run path (repeat for multiple methods)",
    )
    parser.add_argument(
        "--log", action="append", dest="logs", metavar="PATH",
        help="Training log file fallback (repeat for multiple methods)",
    )
    parser.add_argument(
        "--method", action="append", dest="methods", metavar="NAME",
        help="Method name — one per --wandb_run or --log, in order",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=_SAVE_DIR,
        help="Output directory for baselines.csv and acc_curve.png",
    )
    args = parser.parse_args()

    sources = (args.wandb_runs or []) + (args.logs or [])
    if not sources:
        parser.error("Provide at least one --wandb_run or --log argument.")
    if len(sources) != len(args.methods or []):
        parser.error("Number of sources (--wandb_run + --log) must match number of --method args.")

    # Pair each source with its method in order
    wandb_pairs = list(zip(args.wandb_runs or [], args.methods or []))
    remaining_methods = (args.methods or [])[(len(args.wandb_runs or [])):]
    log_pairs = list(zip(args.logs or [], remaining_methods))

    csv_path = args.out_dir / "baselines.csv"
    plot_path = args.out_dir / "acc_curve.png"
    rows: List[dict] = []
    method_records = []

    for run_path, method in wandb_pairs:
        print(f"Fetching W&B run {run_path} ({method}) …")
        try:
            result = fetch_wandb_run(run_path, method)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        _print_and_collect(method, result, run_path, "wandb", rows, method_records)

    for log_str, method in log_pairs:
        log_path = Path(log_str)
        if not log_path.exists():
            print(f"WARNING: log not found: {log_path} — skipping.")
            continue
        records = parse_log(log_path)
        if not records:
            print(f"WARNING: no task-result lines in {log_path} — skipping.")
            continue
        result = extract_final_metrics_from_log(records)
        _print_and_collect(method, result, str(log_path), "log", rows, method_records)

    if rows:
        write_csv(rows, csv_path)
        print(f"Saved CSV  → {csv_path}")
    if method_records:
        plot_acc_curves(method_records, plot_path)


def _print_and_collect(method, result, source, source_type, rows, method_records):
    aa = result["AA"]
    aia = result.get("AIA")
    fm = result.get("FM")
    n = result["n_tasks"]
    aia_str = f"{aia:.2f}" if aia is not None else ""
    fm_str = f"{fm:.2f}" if fm is not None else ""
    print(f"  {method}: AA={aa:.2f}%  AIA={aia_str}  FM={fm_str}  n_tasks={n}  [{source_type}]")
    rows.append({
        "method": method,
        "n_tasks": n,
        "AA": f"{aa:.2f}",
        "AIA": aia_str,
        "FM": fm_str,
        "source": source,
        "eval_date": str(date.today()),
    })
    method_records.append((method, result["records"]))


if __name__ == "__main__":
    main()
