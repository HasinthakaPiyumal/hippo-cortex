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

Main comparison table (Paper 1):
    python experiments/stage1_split_cifar100/eval_baseline.py --table \\
        --wandb_run hasinthakapiyumal-uok/inf-ssm-baseline/d591d88c21e6 --method inf_ssm \\
        --wandb_run hasinthakapiyumal-uok/mamba-cl-baseline/3388e092e0b1 --method mamba_cl_shared_head
Writes results/stage1_split_cifar100/main_table.md and .csv with
AA, AIA, FM, TIL and BWT for each method, plus acc_curve.png.
Add HippoCortex with one more --wandb_run/--method pair; no code change needed.

NOTE on BWT: only available when the run logs the per-task accuracy matrix as
`test/class_<i>_acc` columns (inf-ssm-protocol runs: seq, inf_ssm, mamba_cl_shared_head).
Mamba-CL's own protocol does not log this matrix to W&B, so BWT is None for --method mamba_cl.
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
        "TIL_AA": "TIL",
    },
}

# Fallback for unknown methods — try these common key names
_WANDB_FALLBACK_KEYS = {
    "task": "task",
    "AA": "AA",
    "AIA": "AIA",
    "FM": "FM",
    "TIL_AA": "TIL",
    "class_inc_last_acc": "AA",
    "class_inc_last_forg": "FM",
}

# Per-task accuracy matrix, logged only by inf-ssm-protocol runs (seq, inf_ssm,
# mamba_cl_shared_head): row at task t holds test/class_<i>_acc for i in 0..t.
_CLASS_ACC_RE = re.compile(r"^test/class_(\d+)_acc$")

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
    return {"AA": aa_pct, "AIA": None, "FM": fm_pct, "TIL": None, "BWT": None,
            "n_tasks": task_id, "records": records}


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

    # Scan every row, with no `keys=` filter: W&B returns only rows that hold
    # ALL the keys asked for, so a run that logs other key names would come
    # back empty. Mamba-CL trained with the Inf-SSM code (--cl_method mamba_cl,
    # the shared-head protocol) logs AA/AIA/FM, not class_inc_last_*; rows
    # that miss the method's keys are read with the fallback map instead.
    task_key = "task"
    rows_by_task: Dict[int, Dict[str, float]] = {}
    class_accs_by_task: Dict[int, Dict[int, float]] = {}

    def _map_row(row: dict, mapping: Dict[str, str]) -> Dict[str, float]:
        mapped = {}
        for wk, ck in mapping.items():
            if ck == "task" or wk not in row or row[wk] is None:
                continue
            v = float(row[wk])
            # W&B may log fractions (0–1); convert to percentage
            if v <= 1.0:
                v *= 100.0
            mapped[ck] = v
        return mapped

    for row in run.scan_history():
        if task_key not in row or row[task_key] is None:
            continue
        try:
            task_id = int(row[task_key])
        except (TypeError, ValueError):
            continue
        mapped = _map_row(row, key_map)
        if "AA" not in mapped:
            mapped = _map_row(row, _WANDB_FALLBACK_KEYS)
        if "AA" in mapped:
            rows_by_task[task_id] = mapped  # last row for this task wins

        # Per-task accuracy matrix, if this run logs it (inf-ssm-protocol runs).
        class_row = {}
        for k, v in row.items():
            m = _CLASS_ACC_RE.match(k)
            if m is not None and v is not None:
                class_row[int(m.group(1))] = float(v)
        if class_row:
            class_accs_by_task[task_id] = class_row  # last row wins, same as above

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
        "TIL": last.get("TIL"),
        "BWT": compute_bwt(class_accs_by_task),
        "n_tasks": len(sorted_tasks),
        "records": records,
    }


def compute_bwt(class_accs_by_task: Dict[int, Dict[int, float]]) -> Optional[float]:
    """
    Backward Transfer, in percentage points: mean over tasks i (all but the
    last) of (accuracy on task i after the final task) - (accuracy on task i
    right after training task i).

    class_accs_by_task[t][i] = fraction accuracy on task i after training
    task t (i <= t), as logged under test/class_<i>_acc. Returns None if
    fewer than 2 tasks have a recorded matrix row.
    """
    if len(class_accs_by_task) < 2:
        return None
    sorted_tasks = sorted(class_accs_by_task.keys())
    last_row = class_accs_by_task[sorted_tasks[-1]]
    diffs = []
    for i in sorted_tasks[:-1]:
        diag = class_accs_by_task.get(i, {}).get(i)
        final = last_row.get(i)
        if diag is not None and final is not None:
            diffs.append(final - diag)
    if not diffs:
        return None
    return (sum(diffs) / len(diffs)) * 100.0


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


_PROTOCOL_FOOTNOTE = (
    "Protocol note: Mamba-CL trains about 10M parameters with null-space "
    "projection, while Inf-SSM trains all weights with a shared 100-way head."
)


def write_main_table(table_rows: List[dict], out_dir: Path) -> None:
    """
    Write results/stage1_split_cifar100/main_table.{md,csv} — the Paper 1
    comparison table: one row per method, with AA, AIA, FM, TIL and BWT,
    traceable back to its W&B run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "AA", "AIA", "FM", "TIL", "BWT", "wandb_run", "eval_date"]

    csv_path = out_dir / "main_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Saved table → {csv_path}")

    md_path = out_dir / "main_table.md"
    header = "| Method | AA | AIA | FM | TIL | BWT | W&B run | Date |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in table_rows:
        lines.append(
            f"| {row['method']} | {row['AA']} | {row['AIA']} | {row['FM']} | "
            f"{row['TIL']} | {row['BWT']} | `{row['wandb_run']}` | {row['eval_date']} |"
        )
    lines.append("")
    lines.append(_PROTOCOL_FOOTNOTE)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved table → {md_path}")


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
        help="Method name — one per source, in order. All --wandb_run sources are paired first, then --log sources, regardless of the order typed on the command line.",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=_SAVE_DIR,
        help="Output directory for baselines.csv and acc_curve.png",
    )
    parser.add_argument(
        "--table", action="store_true",
        help="Also write main_table.md and main_table.csv (AA, AIA, FM, TIL, BWT per method)",
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
    table_rows: List[dict] = []

    for run_path, method in wandb_pairs:
        print(f"Fetching W&B run {run_path} ({method}) …")
        try:
            result = fetch_wandb_run(run_path, method)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        _print_and_collect(method, result, run_path, "wandb", rows, method_records, table_rows)

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
        _print_and_collect(method, result, str(log_path), "log", rows, method_records, table_rows)

    if rows:
        write_csv(rows, csv_path)
        print(f"Saved CSV  → {csv_path}")
    if method_records:
        plot_acc_curves(method_records, plot_path)
    if args.table and table_rows:
        write_main_table(table_rows, args.out_dir)


def _print_and_collect(method, result, source, source_type, rows, method_records, table_rows):
    aa = result["AA"]
    aia = result.get("AIA")
    fm = result.get("FM")
    til = result.get("TIL")
    bwt = result.get("BWT")
    n = result["n_tasks"]
    aia_str = f"{aia:.2f}" if aia is not None else ""
    fm_str = f"{fm:.2f}" if fm is not None else ""
    til_str = f"{til:.2f}" if til is not None else ""
    bwt_str = f"{bwt:.2f}" if bwt is not None else ""
    print(f"  {method}: AA={aa:.2f}%  AIA={aia_str}  FM={fm_str}  TIL={til_str}  BWT={bwt_str}  n_tasks={n}  [{source_type}]")
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
    table_rows.append({
        "method": method,
        "AA": f"{aa:.2f}",
        "AIA": aia_str,
        "FM": fm_str,
        "TIL": til_str,
        "BWT": bwt_str,
        "wandb_run": source if source_type == "wandb" else "",
        "eval_date": str(date.today()),
    })


if __name__ == "__main__":
    main()
