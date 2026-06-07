"""
utils/metrics.py
================
Continual-learning evaluation metrics for HippoCortex.

Implements Equations (39)–(41) from Lee et al. (2025), Inf-SSM, Appendix G.2.
Results are directly comparable to that paper without any re-normalisation.

Convention: R[i][j] = accuracy on task j after training through task i (0-based).
R is lower-triangular; shape (T, T).
"""

from __future__ import annotations
from typing import Sequence

Matrix = Sequence[Sequence[float]]


def _validate(R: Matrix) -> int:
    T = len(R)
    if T == 0:
        raise ValueError("Result matrix R must have at least one row.")
    for i, row in enumerate(R):
        if len(row) < i + 1:
            raise ValueError(
                f"Row {i} must have at least {i + 1} entries, got {len(row)}."
            )
    return T


def average_accuracy(R: Matrix) -> float:
    """Eq. (39): AA_k = (1/k) * sum_{j=1}^{k} a_{k,j}

    Average accuracy across all tasks after the final training step.
    Higher is better.
    """
    T = _validate(R)
    return sum(R[T - 1][j] for j in range(T)) / T


def average_incremental_accuracy(R: Matrix) -> float:
    """Eq. (40): AIA_k = (1/k) * sum_{i=1}^{k} AA_i

    Average of AA computed after each task. Captures performance
    throughout training, not only at the end. Higher is better.
    """
    T = _validate(R)
    aa_at_step = [
        sum(R[i][j] for j in range(i + 1)) / (i + 1)
        for i in range(T)
    ]
    return sum(aa_at_step) / T


def forgetting_measure(R: Matrix) -> float:
    """Eq. (41): FM_k = (1/(k-1)) * sum_{j=1}^{k-1} max_{i in 1..k-1}(a_{i,j} - a_{k,j})

    For each previously-seen task, measures the drop from its peak accuracy
    to its final accuracy. Averaged across all old tasks. Lower is better.
    Returns 0.0 when T == 1.
    """
    T = _validate(R)
    if T == 1:
        return 0.0

    total_drop = 0.0
    for j in range(T - 1):
        peak  = max(R[i][j] for i in range(j, T - 1))
        final = R[T - 1][j]
        total_drop += peak - final

    return total_drop / (T - 1)


if __name__ == "__main__":
    R = [
        [0.90, 0.00, 0.00],
        [0.80, 0.85, 0.00],
        [0.70, 0.75, 0.80],
    ]

    aa  = average_accuracy(R)
    aia = average_incremental_accuracy(R)
    fm  = forgetting_measure(R)

    assert abs(aa  - 0.750) < 1e-9, f"AA  mismatch: {aa}"
    assert abs(aia - 0.825) < 1e-9, f"AIA mismatch: {aia}"
    assert abs(fm  - 0.150) < 1e-9, f"FM  mismatch: {fm}"

    print("utils/metrics.py self-test PASSED")
    print(f"  AA  = {aa:.4f}   (expected 0.7500)")
    print(f"  AIA = {aia:.4f}   (expected 0.8250)")
    print(f"  FM  = {fm:.4f}   (expected 0.1500)")