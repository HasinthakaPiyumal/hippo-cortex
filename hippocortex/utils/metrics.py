from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


class ResultMatrix:
    """
    R[i][j] = accuracy on task j's test set after training through task i.

    Only the lower triangle is valid (j <= i).
    Accuracies are fractions in [0, 1], NOT percentages.
    """

    def __init__(self, num_tasks: int) -> None:
        if num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {num_tasks}")
        self.num_tasks = num_tasks
        self._data: List[List[float]] = [
            [float("nan")] * num_tasks for _ in range(num_tasks)
        ]

    def update(self, task_i: int, task_j: int, accuracy: float) -> None:
        if not (0 <= task_i < self.num_tasks):
            raise IndexError(f"task_i={task_i} out of range [0, {self.num_tasks-1}]")
        if not (0 <= task_j <= task_i):
            raise IndexError(
                f"task_j={task_j} must satisfy 0 <= task_j <= task_i={task_i}"
            )
        if not (0.0 <= accuracy <= 1.0):
            warnings.warn(
                f"accuracy={accuracy:.4f} is outside [0, 1]. "
                "Pass a fraction, not a percentage.",
                UserWarning,
                stacklevel=2,
            )
        self._data[task_i][task_j] = float(accuracy)

    def get(self, task_i: int, task_j: int) -> float:
        value = self._data[task_i][task_j]
        if value != value:  # NaN check
            raise ValueError(f"Entry a[{task_i}][{task_j}] has not been set yet.")
        return value

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self._data, dtype=torch.float32)

    def __repr__(self) -> str:
        rows = []
        for i, row in enumerate(self._data):
            formatted = [f"{v:.4f}" if v == v else "  nan " for v in row]
            rows.append(f"  [{', '.join(formatted)}]")
        return "ResultMatrix(\n" + "\n".join(rows) + "\n)"


def compute_aa(R: ResultMatrix, k: int) -> float:
    """
    Average Accuracy after finishing task k (0-indexed).
    Eq. 39: AA_k = (1/(k+1)) * sum of a[k][j] for j in 0..k
    """
    if k >= R.num_tasks:
        raise IndexError(f"k={k} exceeds num_tasks-1={R.num_tasks-1}")
    total = sum(R.get(k, j) for j in range(k + 1))
    return total / (k + 1)


def compute_aia(R: ResultMatrix, k: int) -> float:
    """
    Average Incremental Accuracy up through task k (0-indexed).
    Eq. 40: AIA_k = (1/(k+1)) * sum of AA_i for i in 0..k
    """
    if k >= R.num_tasks:
        raise IndexError(f"k={k} exceeds num_tasks-1={R.num_tasks-1}")
    aa_sum = sum(compute_aa(R, i) for i in range(k + 1))
    return aa_sum / (k + 1)


def compute_fm(R: ResultMatrix, k: int) -> float:
    """
    Forgetting Measure after task k (0-indexed).
    Eq. 41: FM_k = (1/k) * sum over j in 0..k-1 of [max(a[i][j]) - a[k][j]]
    Returns 0.0 for k=0 (no old tasks to forget).
    """
    if k >= R.num_tasks:
        raise IndexError(f"k={k} exceeds num_tasks-1={R.num_tasks-1}")
    if k == 0:
        return 0.0
    total = 0.0
    for j in range(k):
        peak = max(R.get(i, j) for i in range(j, k))
        final = R.get(k, j)
        total += peak - final
    return total / k


def compute_all_metrics(R: ResultMatrix, num_tasks: int) -> Dict[str, List[float]]:
    """Compute AA, AIA, FM for every task index 0..num_tasks-1."""
    if num_tasks > R.num_tasks:
        raise ValueError(f"num_tasks={num_tasks} exceeds R.num_tasks={R.num_tasks}")
    aa_list, aia_list, fm_list = [], [], []
    for k in range(num_tasks):
        aa_list.append(compute_aa(R, k))
        aia_list.append(compute_aia(R, k))
        fm_list.append(compute_fm(R, k))
    return {"aa": aa_list, "aia": aia_list, "fm": fm_list}


@dataclass
class ContinualMetrics:
    """
    Stateful wrapper for the training loop.

    Usage:
        cm = ContinualMetrics(num_tasks=10)
        for task_i in range(10):
            train(task_i)
            for task_j in range(task_i + 1):
                cm.record(task_i, task_j, evaluate(task_j))
        print(cm.summary())
    """
    num_tasks: int
    _R: ResultMatrix = field(init=False, repr=False)
    _current_task: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        self._R = ResultMatrix(self.num_tasks)

    def record(self, task_i: int, task_j: int, accuracy: float) -> None:
        self._R.update(task_i, task_j, accuracy)
        self._current_task = max(self._current_task, task_i)

    def aa(self, k: Optional[int] = None) -> float:
        k = self._current_task if k is None else k
        return compute_aa(self._R, k)

    def aia(self, k: Optional[int] = None) -> float:
        k = self._current_task if k is None else k
        return compute_aia(self._R, k)

    def fm(self, k: Optional[int] = None) -> float:
        k = self._current_task if k is None else k
        return compute_fm(self._R, k)

    def summary(self) -> Dict[str, object]:
        k = self._current_task
        if k < 0:
            raise RuntimeError("No data recorded yet. Call record() first.")
        all_metrics = compute_all_metrics(self._R, k + 1)
        return {
            "num_tasks_completed": k + 1,
            "aa_per_task":  all_metrics["aa"],
            "aia_per_task": all_metrics["aia"],
            "fm_per_task":  all_metrics["fm"],
            "final_aa":     all_metrics["aa"][k],
            "final_aia":    all_metrics["aia"][k],
            "final_fm":     all_metrics["fm"][k],
        }

    @property
    def result_matrix(self) -> ResultMatrix:
        return self._R
