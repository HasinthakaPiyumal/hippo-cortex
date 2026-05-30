"""
utils/metrics.py
================
Continual-learning evaluation metrics for HippoCortex.

All three functions implement **exactly** the formulas from:

    Lee et al. (2025). "Exemplar-Free Continual Learning for State Space Models."
    arXiv:2505.18604, Appendix G.2, Equations (39)–(41).

This ensures our reported numbers are directly comparable to the Inf-SSM
paper without any re-normalisation or definitional drift.

Result matrix convention
------------------------
R[i][j]  =  accuracy on task j's test set **after** the model has been
             trained through task i  (0-based Python indices).

R is lower-triangular: R[i][j] is only defined for j <= i.
The matrix has shape (T, T) where T is the total number of tasks.

Paper uses 1-based indices; we map  paper's k → Python k-1,
paper's a_{k,j} → R[k-1][j-1].  All docstrings quote the paper formula
first, then show the Python translation.
"""

from __future__ import annotations
from typing import Sequence


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Matrix = Sequence[Sequence[float]]   # R[i][j], lower-triangular, 0-based


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _validate(R: Matrix) -> int:
    """Return T (number of tasks) and sanity-check shape."""
    T = len(R)
    if T == 0:
        raise ValueError("Result matrix R must have at least one row.")
    for i, row in enumerate(R):
        if len(row) < i + 1:
            raise ValueError(
                f"Row {i} must have at least {i + 1} entries "
                f"(lower-triangular), got {len(row)}."
            )
    return T


# ---------------------------------------------------------------------------
# Eq. (39) — Average Accuracy
# ---------------------------------------------------------------------------

def average_accuracy(R: Matrix) -> float:
    """Average Accuracy (AA) after training on all T tasks.

    Paper formula (Eq. 39), 1-based:
        AA_k = (1/k) * sum_{j=1}^{k} a_{k,j}

    Called with k = T (final task), this is the standard end-of-sequence
    snapshot of accuracy across all tasks.

    Interpretation
    --------------
    Higher is better.  AA answers: "After the model has seen every task,
    how well does it perform on each of them on average?"  Low AA reveals
    that the model has forgotten earlier tasks.

    Parameters
    ----------
    R : 2-D array-like, lower-triangular, 0-based
        R[i][j] = accuracy on task j after training through task i.
        Only the last row (i = T-1) is used here.

    Returns
    -------
    float
        AA in the same unit as the entries of R (e.g. 0–100 for %).

    Example
    -------
    >>> R = [[0.9, 0,   0  ],
    ...      [0.8, 0.85, 0 ],
    ...      [0.7, 0.75, 0.8]]
    >>> round(average_accuracy(R), 4)
    0.75
    """
    T = _validate(R)
    last_row = R[T - 1]          # a_{T, j}  for j in 1..T  (paper, 1-based)
    return sum(last_row[j] for j in range(T)) / T


# ---------------------------------------------------------------------------
# Eq. (40) — Average Incremental Accuracy
# ---------------------------------------------------------------------------

def average_incremental_accuracy(R: Matrix) -> float:
    """Average Incremental Accuracy (AIA) over the full task sequence.

    Paper formula (Eq. 40), 1-based:
        AIA_k = (1/k) * sum_{i=1}^{k} AA_i

    where AA_i = (1/i) * sum_{j=1}^{i} a_{i,j}   (Eq. 39 at step i).

    Interpretation
    --------------
    Higher is better.  AIA averages the *trajectory* of AA values
    measured after each task, not just the final one.  A method that
    starts high and degrades will have lower AIA than AA; a method that
    ramps up will have higher AIA than AA.  AIA penalises forgetting
    throughout training, not only at the end.

    Parameters
    ----------
    R : 2-D array-like, lower-triangular, 0-based
        R[i][j] = accuracy on task j after training through task i.
        All T rows are used.

    Returns
    -------
    float
        AIA in the same unit as the entries of R.

    Example
    -------
    >>> R = [[0.9, 0,   0  ],
    ...      [0.8, 0.85, 0 ],
    ...      [0.7, 0.75, 0.8]]
    >>> # AA_0 = 0.9, AA_1 = (0.8+0.85)/2 = 0.825, AA_2 = (0.7+0.75+0.8)/3 = 0.75
    >>> # AIA  = (0.9 + 0.825 + 0.75) / 3
    >>> round(average_incremental_accuracy(R), 6)
    0.825
    """
    T = _validate(R)

    # AA_i  (paper, 1-based i)  =  average of row i-1 up to column i-1 (0-based)
    aa_at_step = [
        sum(R[i][j] for j in range(i + 1)) / (i + 1)
        for i in range(T)
    ]
    return sum(aa_at_step) / T


# ---------------------------------------------------------------------------
# Eq. (41) — Forgetting Measure
# ---------------------------------------------------------------------------

def forgetting_measure(R: Matrix) -> float:
    """Forgetting Measure (FM) at the end of the task sequence.

    Paper formula (Eq. 41), 1-based:
        FM_k = (1 / (k-1)) * sum_{j=1}^{k-1}  max_{i in 1..k-1} (a_{i,j} - a_{k,j})

    Interpretation
    --------------
    Lower is better (0 = no forgetting).  For each previously-seen task j,
    FM finds the **peak accuracy** that task ever achieved during training
    (over all intermediate checkpoints), subtracts the **final accuracy**
    on that task, and averages those drops.

    The max is taken only up to step k-1 (not including step k itself)
    because the model's performance on task j when it was *most recently*
    trained is the natural reference point for "how good we used to be".
    This convention matches the Inf-SSM paper exactly.

    A negative per-task drop is possible (final accuracy higher than any
    intermediate checkpoint, e.g. via positive backward transfer); in
    practice, FM is computed with the raw max so backward transfer shows
    up as a negative contribution, pulling FM down.

    Parameters
    ----------
    R : 2-D array-like, lower-triangular, 0-based
        R[i][j] = accuracy on task j after training through task i.

    Returns
    -------
    float
        FM in the same unit as the entries of R.
        Undefined (returns 0.0) when T == 1 (no forgetting can be measured
        with a single task).

    Raises
    ------
    ValueError
        If the matrix is malformed (see _validate).

    Example
    -------
    >>> R = [[0.9, 0,   0  ],
    ...      [0.8, 0.85, 0 ],
    ...      [0.7, 0.75, 0.8]]
    >>> # Task 0: peak in rows 0..1 = max(R[0][0], R[1][0]) = max(0.9, 0.8) = 0.9
    >>> #         final = R[2][0] = 0.7  →  drop = 0.9 - 0.7 = 0.2
    >>> # Task 1: peak in rows 1..1 = R[1][1] = 0.85
    >>> #         final = R[2][1] = 0.75 →  drop = 0.85 - 0.75 = 0.1
    >>> # FM = (0.2 + 0.1) / 2 = 0.15
    >>> round(forgetting_measure(R), 4)
    0.15
    """
    T = _validate(R)

    if T == 1:
        return 0.0  # no previous tasks to forget

    total_drop = 0.0
    for j in range(T - 1):           # task j in 0..T-2  (paper: j=1..k-1)
        # Peak accuracy on task j over all steps *before* the final step.
        # Paper: max_{i in 1..k-1} a_{i,j}  →  rows j..T-2 in 0-based.
        # (Row j is the earliest row where task j appears on the diagonal.)
        peak = max(R[i][j] for i in range(j, T - 1))
        final = R[T - 1][j]           # a_{k,j} in the paper
        total_drop += peak - final

    return total_drop / (T - 1)


# ---------------------------------------------------------------------------
# Self-test (run: python -m utils.metrics)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 3-task worked example — all values in [0, 1]
    #
    #        T0    T1    T2
    # After T0: [0.90]
    # After T1: [0.80, 0.85]
    # After T2: [0.70, 0.75, 0.80]
    #
    # R stored as a full 3×3 lower-triangular matrix; off-diagonal upper
    # entries are 0 and never read.

    R = [
        [0.90, 0.00, 0.00],
        [0.80, 0.85, 0.00],
        [0.70, 0.75, 0.80],
    ]

    aa  = average_accuracy(R)
    aia = average_incremental_accuracy(R)
    fm  = forgetting_measure(R)

    # --- expected values (hand-computed) ---
    # AA  = (0.70 + 0.75 + 0.80) / 3            = 0.75
    # AA0 = 0.90 / 1                             = 0.900
    # AA1 = (0.80 + 0.85) / 2                   = 0.825
    # AA2 = (0.70 + 0.75 + 0.80) / 3            = 0.750
    # AIA = (0.900 + 0.825 + 0.750) / 3         = 0.825
    # FM:
    #   task 0 drop = max(R[0][0], R[1][0]) - R[2][0] = 0.90 - 0.70 = 0.20
    #   task 1 drop = max(R[1][1])          - R[2][1] = 0.85 - 0.75 = 0.10
    #   FM = (0.20 + 0.10) / 2                = 0.150

    assert abs(aa  - 0.750) < 1e-9, f"AA  mismatch: {aa}"
    assert abs(aia - 0.825) < 1e-9, f"AIA mismatch: {aia}"
    assert abs(fm  - 0.150) < 1e-9, f"FM  mismatch: {fm}"

    print("utils/metrics.py self-test PASSED")
    print(f"  AA  = {aa:.4f}   (expected 0.7500)")
    print(f"  AIA = {aia:.4f}   (expected 0.8250)")
    print(f"  FM  = {fm:.4f}   (expected 0.1500)")