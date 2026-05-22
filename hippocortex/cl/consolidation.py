"""
Consolidation driver — the SWR sleep phase.

OWNER: Hasinthaka Piyumal

Called once per task, after the awake phase. Orchestrates:
  1. Sample Ĥ_past from SWRGenerator for every previously seen task.
  2. Compute ELBO loss on the synthetic hidden states.
  3. Route all gradients through NullSpaceProjector before applying.
  4. After gradient update, call StatsBuffer.update() and NullSpaceProjector.update()
     with the current task's real hidden states.

Implementation notes:
- n_samples controls how many synthetic samples to draw per prior task.
  Default 512 balances quality of replay vs. compute cost.
- Returns the mean replay loss (float) for logging purposes.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer

from hippocortex.models.backbone import MambaBackbone
from hippocortex.models.swr_generator import SWRGenerator
from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.cl.null_space_projector import NullSpaceProjector


def consolidate(
    backbone: MambaBackbone,
    swr_gen: SWRGenerator,
    projector: NullSpaceProjector,
    buffer: StatsBuffer,
    task_id: int,
    optimizer: Optimizer,
    current_hidden_states: Tensor,
    n_samples: int = 512,
) -> float:
    """
    Run the sleep/SWR consolidation phase for the task that just completed.

    Args:
        backbone:              The Mamba model being trained.
        swr_gen:               The conditional VAE SWR generator.
        projector:             The null-space gradient projector.
        buffer:                The per-task stats buffer.
        task_id:               Index of the task that just finished awake training.
        optimizer:             Shared optimizer for backbone + swr_gen.
        current_hidden_states: Hidden states collected during the awake phase
                               of task_id, shape (N, d_model). Used to update
                               the buffer and projector AFTER gradient step.
        n_samples:             Synthetic samples to draw per prior task.

    Returns:
        Mean replay loss (float) for logging.
    """
    raise NotImplementedError
