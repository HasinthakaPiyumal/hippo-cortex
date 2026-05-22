"""
StatsBuffer — per-task compact memory store.

OWNER: Hasinthaka Piyumal

This is the core paper claim: memory grows as O(T × d_model) regardless of
how many samples were seen. For T=20 tasks and d_model=128 float32, total
footprint is 20 × 128 × 2 × 4 = 20 KB.

Implementation notes:
- update() receives a batch of real hidden states for a completed task,
  computes their mean and variance, and stores (µ, σ²) indexed by task_id.
- get_stats() retrieves stored statistics for replay.
- memory_bytes() must return the true byte count (used in the paper's
  memory-vs-task-count plot).
"""
from __future__ import annotations

import torch
from torch import Tensor


class StatsBuffer:
    def __init__(self) -> None:
        self._store: dict[int, tuple[Tensor, Tensor]] = {}

    def update(self, task_id: int, hidden_states: Tensor) -> None:
        """
        Compute and store (µ, σ²) for task_id.

        Args:
            task_id:       Integer task index (0-based).
            hidden_states: Shape (N, d_model) — a batch of hidden states collected
                           across the training set of the completed task.
        """
        raise NotImplementedError

    def get_stats(self, task_id: int) -> tuple[Tensor, Tensor]:
        """
        Return stored (µ, σ²) for task_id.

        Returns:
            mu:     shape (d_model,)
            sigma2: shape (d_model,)

        Raises:
            KeyError if task_id has not been stored yet.
        """
        raise NotImplementedError

    def task_ids(self) -> list[int]:
        """Return sorted list of all stored task ids."""
        return sorted(self._store.keys())

    def memory_bytes(self) -> int:
        """
        Return exact byte count of all stored tensors.
        Must satisfy: bytes == len(task_ids) * d_model * 2 * element_size.
        """
        raise NotImplementedError
