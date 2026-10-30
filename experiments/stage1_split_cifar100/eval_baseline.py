"""
Baseline evaluation script — produces results/stage1_split_cifar100/baselines.csv.

OWNER: Thagya Kavindi

Usage:
    python experiments/stage1_split_cifar100/eval_baseline.py \
        --log path/to/mamba_cl_run.log --method mamba_cl

    # Multiple baselines in one call:
    python experiments/stage1_split_cifar100/eval_baseline.py \
        --log logs/mamba_cl_large.log --method mamba_cl \
        --log logs/inf_ssm_base.log   --method inf_ssm

Output CSV columns:
    method, n_tasks, AA, FM, memory_mb, log_file, eval_date

 NOTE: Per-task accuracy matrix is NOT in the Mamba-CL log by default (only the
running average is printed).  BWT therefore cannot be computed from logs alone.
Tell Hasinthaka to add a matrix-print line to train_eval.py if BWT is needed.
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SAVE_DIR = Path("results/stage1_split_cifar100")

# Matches lines like:
# :: ** Results of task [3]: [ class_inc_last_acc: 91.20% | class_inc_last_forg:  1.85% ] **
_TASK_LINE_RE = re.compile(
    r"Results of task \[(\d+)\].*?"
    r"class_inc_last_acc:\s*([\d.]+)%.*?"
    r"class_inc_last_forg:\s*([\d.]+)%"
)


def parse_log(log_path: Path) -> List[Tuple[int, float, float]]:
    """
    Return [(task_id, acc_pct, forg_pct), ...] in log order.

    task_id is 1-indexed (as printed).
    acc and forg are percentages, e.g. 91.20 not 0.9120.
    """
    records: List[Tuple[int, float, float]] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in _TASK_LINE_RE.finditer(text):
        records.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return records


def extract_final_metrics(
    records: List[Tuple[int, float, float]],
) -> Tuple[float, float, int]:
    """Return (AA%, FM%, n_tasks) from the last log entry."""
    if not records:
        raise ValueError("No task-result lines found in the log.")
    task_id, aa_pct, fm_pct = records[-1]
    return aa_pct, fm_pct, task_id


def write_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "n_tasks", "AA", "FM", "memory_mb", "log_file", "eval_date"]
    existing: List[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    # Overwrite any row for the same method+log_file pair
    existing = [
        r for r in existing
        if not any(
            r.get("method") == row["method"] and r.get("log_file") == row["log_file"]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Mamba-CL / Inf-SSM training logs → CSV + accuracy-curve plot."
    )
    parser.add_argument(
        "--log", action="append", dest="logs", metavar="PATH",
        help="Training log file (repeat once per method)",
    )
    parser.add_argument(
        "--method", action="append", dest="methods", metavar="NAME",
        help="Method name for the corresponding --log",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=_SAVE_DIR,
        help="Output directory for baselines.csv and acc_curve.png",
    )
    args = parser.parse_args()

    if not args.logs:
        parser.error("Provide at least one --log … --method … pair.")
    if len(args.logs) != len(args.methods):
        parser.error("--log and --method counts must match.")

    csv_path = args.out_dir / "baselines.csv"
    plot_path = args.out_dir / "acc_curve.png"

    rows: List[dict] = []
    method_records: List[Tuple[str, List[Tuple[int, float, float]]]] = []

    for log_str, method in zip(args.logs, args.methods):
        log_path = Path(log_str)
        if not log_path.exists():
            print(f"WARNING: log not found: {log_path} — skipping.")
            continue

        records = parse_log(log_path)
        if not records:
            print(f"WARNING: no task-result lines in {log_path} — skipping.")
            continue

        aa_pct, fm_pct, n_tasks = extract_final_metrics(records)
        print(f"{method}: AA={aa_pct:.2f}%  FM={fm_pct:.2f}%  n_tasks={n_tasks}  ({log_path.name})")

        rows.append({
            "method": method,
            "n_tasks": n_tasks,
            "AA": f"{aa_pct:.2f}",
            "FM": f"{fm_pct:.2f}",
            "memory_mb": "",
            "log_file": str(log_path),
            "eval_date": str(date.today()),
        })
        method_records.append((method, records))

    if rows:
        write_csv(rows, csv_path)
        print(f"Saved CSV  → {csv_path}")

    if method_records:
        plot_acc_curves(method_records, plot_path)


if __name__ == "__main__":
    main()
