"""
Continual learning evaluation metrics.

acc_matrix[i, j] = accuracy on task j after training on task i.
Shape: (n_tasks, n_tasks). Upper triangle (j > i) is undefined.

Formulas follow Lange et al. 2021 (A Continual Learning Survey), Appendix.
"""
import numpy as np


def average_accuracy(acc_matrix: np.ndarray) -> float:
    """Mean diagonal accuracy after training all T tasks."""
    raise NotImplementedError


def average_forgetting(acc_matrix: np.ndarray) -> float:
    """Mean drop in accuracy on old tasks after learning new ones."""
    raise NotImplementedError


def backward_transfer(acc_matrix: np.ndarray) -> float:
    """Average influence of learning task i on task j < i."""
    raise NotImplementedError


def build_acc_matrix(n_tasks: int) -> np.ndarray:
    """Return an (n_tasks, n_tasks) matrix initialised to -1 (undefined)."""
    return np.full((n_tasks, n_tasks), -1.0, dtype=np.float32)
