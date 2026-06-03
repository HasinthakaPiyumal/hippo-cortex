"""
MambaBackbone — selective SSM encoder + classification head.

OWNER: Praveen Dedigama

Implementation notes:
- Wrap mamba_ssm.Mamba blocks into an L-layer stack.
- Register a forward hook on layer index (n_layers - 1) to capture hidden states.
- extract_hidden() returns the mean-pooled output of that layer, shape (B, d_model).
  This extraction point matches Mamba-CL (Cheng et al. 2025, Section 3.2).
- forward() returns (logits, hidden_states) so the training loop can feed
  hidden_states directly to StatsBuffer and SWRGenerator.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from mamba_ssm import Mamba


class MambaBackbone(nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_state: int, n_classes: int):
        """
        Args:
            d_model:   Hidden dimension of each Mamba SSM block.
            n_layers:  Number of stacked Mamba blocks.
            d_state:   SSM state expansion factor (d_state in mamba_ssm).
            n_classes: Output classes for the current task head.
        """
        super().__init__()
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=d_state) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor, shape (B, T, C) — sequence of patch embeddings or flattened pixels.

        Returns:
            logits:        (B, n_classes)
            hidden_states: (B, d_model) — mean-pooled output of last SSM layer.
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        hidden_states = h.mean(dim=1)
        logits = self.head(hidden_states)
        return logits, hidden_states

    def extract_hidden(self, x: Tensor) -> Tensor:
        """
        Forward pass returning only the hidden states from the penultimate SSM layer.
        Used by StatsBuffer during consolidation.

        Returns:
            (B, d_model)
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return h.mean(dim=1)

    def set_task_head(self, n_classes: int) -> None:
        """Replace the classification head for a new task (task-incremental setting)."""
        d_model = self.head.in_features
        self.head = nn.Linear(d_model, n_classes).to(next(self.parameters()).device)
