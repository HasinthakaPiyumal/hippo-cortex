"""
NullSpaceProjector — gradient orthogonaliser for continual learning.

OWNER: Praveen Dedigama

Implementation notes:
- Maintains a basis matrix U of shape (d_model, k) where k ≤ rank_budget.
- update() adds new task directions via incremental SVD on a batch of hidden states.
  Use torch.linalg.svd; retain the top-r singular vectors where r is chosen to
  capture 99% of variance (or until rank_budget is hit).
- project(grad) computes P = I − U @ U.T and applies it to grad.
  Pure batched PyTorch — no Python loops over individual parameters.

References:
  Saha et al. 2021, Algorithm 1 — GPM (papers/stage1-swr-replay/)
  Cheng et al. 2025, Section 3.2 — Mamba-CL null-space (papers/stage1-swr-replay/)
"""
from __future__ import annotations

import torch
from torch import Tensor


class NullSpaceProjector:
    def __init__(self, rank_budget: int = 200) -> None:
        """
        Args:
            rank_budget: Maximum number of basis vectors to retain across all tasks.
        """
        self.rank_budget = rank_budget
        self._U: Tensor | None = None  # shape (d_model, k), k ≤ rank_budget

    def update(self, hidden_states: Tensor) -> None:
        """
        Extend the basis U with new task's feature directions.

        Args:
            hidden_states: Shape (N, d_model) — hidden states from the completed task.

        Updates self._U in-place (appends new singular vectors, then truncates to
        rank_budget if necessary).
        """
        _, S, Vt = torch.linalg.svd(hidden_states, full_matrices=False)
        var = S ** 2
        cumulative_ratio = torch.cumsum(var, dim=0) / var.sum()
        r = int((cumulative_ratio < 0.99).sum().item()) + 1
        r = min(r, Vt.shape[0])
        new_dirs = Vt[:r].T
        if self._U is None:
            combined = new_dirs
        else:
            combined = torch.cat([self._U, new_dirs], dim=1)
        Q, _ = torch.linalg.qr(combined)
        k_new = min(Q.shape[1], self.rank_budget)
        self._U = Q[:, :k_new]

    def project(self, grad: Tensor) -> Tensor:
        """
        Apply the null-space projection P = I − U @ U.T to a gradient tensor.

        Args:
            grad: Gradient of the same dimension as d_model, any shape (..., d_model).

        Returns:
            Projected gradient with same shape as input.
            If no tasks have been seen yet (U is None), returns grad unchanged.
        """
        if self._U is None:
            return grad
        return grad - (grad @ self._U) @ self._U.T

    @property
    def current_rank(self) -> int:
        """Number of basis vectors currently stored."""
        return 0 if self._U is None else self._U.shape[1]
