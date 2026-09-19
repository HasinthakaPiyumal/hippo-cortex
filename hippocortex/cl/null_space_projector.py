"""
NullSpaceProjector — gradient orthogonaliser for continual learning.

OWNER: Praveen Dedigama

Implementation notes:
- Maintains a basis matrix U of shape (d_model, k) where k <= rank_budget.
- update() adds new task directions using SVD on hidden states.
- Retains directions needed to capture approximately 99% of feature energy.
- project() removes gradient components lying in the protected subspace.

References:
  Saha et al. 2021 — Gradient Projection Memory (GPM)
  Cheng et al. — Mamba-CL null-space continual learning
"""

from __future__ import annotations

import torch
from torch import Tensor


class NullSpaceProjector:
    def __init__(self, rank_budget: int = 200) -> None:
        """
        Args:
            rank_budget:
                Maximum number of protected basis directions
                retained across tasks.
        """
        self.rank_budget = rank_budget

        # Protected basis.
        #
        # Shape:
        #   (d_model, k)
        #
        # where:
        #   d_model = feature dimension
        #   k       = number of protected directions
        self._U: Tensor | None = None

    def update(self, hidden_states: Tensor) -> None:
        """
        Extend the protected basis using hidden states from a completed task.

        Args:
            hidden_states:
                Tensor with shape (N, d_model)

                N       = number of samples
                d_model = number of hidden features

        The method:
            1. Runs SVD on hidden states.
            2. Measures the energy of each feature-space direction.
            3. Keeps enough directions to explain approximately 99% of energy.
            4. Combines them with previously protected directions.
            5. Uses QR to form an orthonormal basis.
            6. Enforces rank_budget.
        """

        # Step 1: Find feature-space directions using SVD.
        _, S, Vt = torch.linalg.svd(
            hidden_states,
            full_matrices=False,
        )

        # Step 2: Convert singular values to energy / variance.
        var = S ** 2
        total_var = var.sum()

        # Step 3: Handle zero-information hidden states.
        if total_var <= torch.finfo(var.dtype).eps:
            return

        # Step 4: Calculate cumulative explained energy.
        cumulative_ratio = torch.cumsum(
            var,
            dim=0,
        ) / total_var

        # Step 5: Select enough directions to explain ~99% of energy.
        r = int(
            (cumulative_ratio < 0.99).sum().item()
        ) + 1

        r = min(
            r,
            Vt.shape[0],
        )

        # Vt stores feature-space directions as rows.
        # Transpose them so directions are stored as columns.
        new_dirs = Vt[:r].T

        # Step 6: Combine previous and new protected directions.
        if self._U is None:
            combined = new_dirs
        else:
            combined = torch.cat(
                [self._U, new_dirs],
                dim=1,
            )

        # Step 7: Orthonormalise the combined basis.
        Q, _ = torch.linalg.qr(combined)

        # Step 8: Respect the maximum protected rank.
        k_new = min(
            Q.shape[1],
            self.rank_budget,
        )

        self._U = Q[:, :k_new]

    def project(self, grad: Tensor) -> Tensor:
        """
        Remove gradient components lying in the protected subspace.

        Args:
            grad:
                Gradient tensor whose final dimension must match
                the feature dimension represented by U.

        Returns:
            Projected gradient with the same shape as grad.
        """

        # No previous protected directions yet.
        if self._U is None:
            return grad

        # Ensure gradient and protected basis use the same feature dimension.
        if grad.shape[-1] != self._U.shape[0]:
            raise ValueError(
                "Gradient feature dimension does not match protected basis dimension: "
                f"grad.shape[-1]={grad.shape[-1]}, "
                f"U.shape[0]={self._U.shape[0]}"
            )

        # Component of the gradient lying inside the protected subspace.
        protected_component = (
            grad @ self._U
        ) @ self._U.T

        # Remove the protected component.
        projected_grad = grad - protected_component

        return projected_grad

    @property
    def current_rank(self) -> int:
        """
        Number of protected basis directions currently stored.
        """
        if self._U is None:
            return 0

        return self._U.shape[1]