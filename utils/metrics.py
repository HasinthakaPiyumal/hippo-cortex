"""
hippocortex/utils/metrics.py
────────────────────────────
Continual-learning evaluation metrics used throughout the HippoCortex project.

All three functions implement the **exact** formulas from:
    Lee et al. (2025) "Exemplar-Free Continual Learning for State Space Models"
    (Inf-SSM), Appendix G.2, Equations (39)–(41).

Using these definitions verbatim ensures our numbers are directly
comparable to the Inf-SSM paper tables (Tab. 1, 4, 5 in that work).

──────────────────────────────────────────────────────────────────────
Notation (mirrors the paper exactly)
──────────────────────────────────────────────────────────────────────
  T          : total number of sequential tasks
  a_{k,j}   : accuracy on task j's test set *after* the model has been
               trained on task k  (j ≤ k, 1-indexed in the paper)
  R          : T×T upper-triangular NumPy array (or nested list) where
               R[i][j] = a_{i+1, j+1}  (0-indexed Python ↔ 1-indexed paper)
               R[i][j] is only meaningful when j ≤ i.

Key metrics (all reported as *percentages* when the inputs are in %):
  AA_k   = (1/k) Σ_{j=1}^{k} a_{k,j}               — Eq. (39)
  AIA_k  = (1/k) Σ_{i=1}^{k} AA_i                   — Eq. (40)
  FM_k   = 1/(k-1) Σ_{j=1}^{k-1} max_{i∈{1..k-1}} (a_{i,j} − a_{k,j})
                                                       — Eq. (41)

The three public functions each take the full R matrix and return the
*final* value (at k = T), which is what all tables in the paper report.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Internal helper
# ──────────────────────────────────────────────────────────────────────

def _to_array(R) -> np.ndarray:
    """Convert R to a float64 NumPy array, accepting lists or arrays."""
    arr = np.array(R, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"R must be a 2-D matrix, got shape {arr.shape}")
    T = arr.shape[0]
    if arr.shape[1] < T:
        raise ValueError(
            f"R must have at least {T} columns (one per task); "
            f"got shape {arr.shape}"
        )
    return arr


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def average_accuracy(R) -> float:
    """
    Average Accuracy (AA) after all tasks — Eq. (39) evaluated at k = T.

        AA_T = (1/T) Σ_{j=1}^{T} a_{T,j}

    In Python 0-indexed terms: mean of the **last row** of R.

    Parameters
    ----------
    R : array-like, shape (T, T)
        R[i][j] = accuracy on task j after training on task i.
        Only the lower-triangular part (j ≤ i) is used.

    Returns
    -------
    float
        AA_T in the same units as the entries of R (typically %).
    """
    R = _to_array(R)
    T = R.shape[0]

    # Last row gives a_{T,1} … a_{T,T}  (0-indexed: R[T-1][0..T-1])
    aa_T = np.mean(R[T - 1, :T])          # Eq. (39) at k = T
    return float(aa_T)


def average_incremental_accuracy(R) -> float:
    """
    Average Incremental Accuracy (AIA) — Eq. (40) evaluated at k = T.

        AIA_T = (1/T) Σ_{k=1}^{T} AA_k

    where   AA_k = (1/k) Σ_{j=1}^{k} a_{k,j}     [Eq. (39)]

    AIA captures overall performance *across the entire task sequence*,
    not just the final snapshot.  This is the metric that rewards a model
    for staying accurate right from task 1.

    Parameters
    ----------
    R : array-like, shape (T, T)

    Returns
    -------
    float
        AIA_T in the same units as R.
    """
    R = _to_array(R)
    T = R.shape[0]

    aa_values = np.empty(T)
    for k in range(T):                     # k is 0-indexed; paper uses k+1
        # AA_{k+1} = mean of R[k][0..k]
        aa_values[k] = np.mean(R[k, : k + 1])   # Eq. (39)

    aia_T = np.mean(aa_values)             # Eq. (40) at k = T
    return float(aia_T)


def forgetting_measure(R) -> float:
    """
    Forgetting Measure (FM) — Eq. (41) evaluated at k = T.

        FM_T = 1/(T-1) Σ_{j=1}^{T-1} max_{i ∈ {1,...,T-1}} (a_{i,j} − a_{T,j})

    For each old task j, FM measures the biggest accuracy drop relative
    to the best checkpoint *before* the final task.  A model with perfect
    knowledge retention has FM = 0.

    Parameters
    ----------
    R : array-like, shape (T, T)

    Returns
    -------
    float
        FM_T ≥ 0 in the same units as R.  Higher means more forgetting.

    Raises
    ------
    ValueError
        If T < 2 (forgetting is undefined for a single task).
    """
    R = _to_array(R)
    T = R.shape[0]

    if T < 2:
        raise ValueError(
            "Forgetting Measure requires at least 2 tasks; got T = 1."
        )

    forgetting_per_task = np.empty(T - 1)
    for j in range(T - 1):                # j is 0-indexed; paper uses j+1
        # Best accuracy on task j+1 seen at any checkpoint i ∈ {1..T-1}
        # 0-indexed: R[0..T-2][j]
        best_before_final = np.max(R[: T - 1, j])   # max over i ∈ {0..T-2}

        # Final accuracy on task j+1: R[T-1][j]
        final = R[T - 1, j]

        forgetting_per_task[j] = best_before_final - final   # Eq. (41) term

    fm_T = np.mean(forgetting_per_task)   # Eq. (41)
    return float(fm_T)


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper — returns all three at once
# ──────────────────────────────────────────────────────────────────────

def compute_all(R) -> dict[str, float]:
    """
    Compute AA, AIA, and FM in a single call.

    Returns
    -------
    dict with keys "AA", "AIA", "FM".
    """
    return {
        "AA":  average_accuracy(R),
        "AIA": average_incremental_accuracy(R),
        "FM":  forgetting_measure(R),
    }


# ──────────────────────────────────────────────────────────────────────
# Quick self-test (run with:  python -m hippocortex.utils.metrics)
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import textwrap

    # --------------- toy example (3 tasks) ---------------
    # After task 1: only a_{1,1} is meaningful
    # After task 2: a_{2,1}, a_{2,2}
    # After task 3: a_{3,1}, a_{3,2}, a_{3,3}
    #
    # R is upper-triangular in the *paper* sense (j ≤ k);
    # here we use 0-indexed rows = trained-up-to-task, cols = evaluated-task.
    #
    # Made-up numbers to sanity-check the formulas manually:
    #   After task 1: task-1 acc = 80
    #   After task 2: task-1 acc = 65  (forgot 15 pts), task-2 acc = 75
    #   After task 3: task-1 acc = 55  (forgot more), task-2 acc = 60, task-3 acc = 70
    R = np.array([
        [80,  0,  0],   # trained on task 1 only
        [65, 75,  0],   # trained on tasks 1-2
        [55, 60, 70],   # trained on tasks 1-3
    ], dtype=float)

    # ---- manual verification ----
    # AA_1 = 80/1 = 80
    # AA_2 = (65+75)/2 = 70
    # AA_3 = (55+60+70)/3 = 61.667
    # AIA_3 = (80 + 70 + 61.667) / 3 = 70.556
    # FM:
    #   j=1 (task 1): max(R[0,0], R[1,0]) - R[2,0] = max(80,65) - 55 = 25
    #   j=2 (task 2): max(R[0,1], R[1,1]) - R[2,1] = max(0, 75) - 60 = 15
    #   FM_3 = (25 + 15) / 2 = 20

    metrics = compute_all(R)
    print(textwrap.dedent(f"""
    ─── Self-test (3-task toy example) ───
    AA  = {metrics['AA']:.3f}   (expected 61.667)
    AIA = {metrics['AIA']:.3f}  (expected 70.556)
    FM  = {metrics['FM']:.3f}   (expected 20.000)
    ──────────────────────────────────────
    """))

    # Assert
    assert abs(metrics["AA"]  - 61.667) < 0.001, "AA mismatch"
    assert abs(metrics["AIA"] - 70.556) < 0.001, "AIA mismatch"
    assert abs(metrics["FM"]  - 20.000) < 0.001, "FM mismatch"
    print("All assertions passed ✓")