"""
Trainer — orchestrates the 3-phase continual learning loop.

OWNER: Hasinthaka Piyumal

Three phases per task:
  1. Warmup:       Supervised CE on current task only. No replay, no projection.
  2. Joint:        CE on current task + ELBO on SWR-generated past.
                   All backbone gradients projected through NullSpaceProjector.
  3. Consolidation: Call consolidate() to update StatsBuffer and NullSpaceProjector.

evaluate() fills one row of the acc_matrix after each task completes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hippocortex.models.backbone import MambaBackbone
from hippocortex.models.swr_generator import SWRGenerator
from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.cl.null_space_projector import NullSpaceProjector
from hippocortex.cl.consolidation import consolidate
from hippocortex.utils.logging import get_logger

_log = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        backbone: MambaBackbone,
        swr_gen: SWRGenerator,
        projector: NullSpaceProjector,
        buffer: StatsBuffer,
        cfg: DictConfig,
    ) -> None:
        raise NotImplementedError

    def train_task(self, task_id: int, loader: DataLoader) -> dict[str, float]:
        """
        Run all three phases for one task.

        Args:
            task_id: Integer task index (0-based).
            loader:  Training DataLoader for this task.

        Returns:
            Dict with keys: "loss_warmup", "loss_joint", "loss_consolidation",
            "replay_loss", "task_id".
        """
        raise NotImplementedError

    def evaluate(self, task_id: int, loader: DataLoader) -> float:
        """
        Evaluate the backbone on a single task's test loader.

        Args:
            task_id: Task to evaluate (used for selecting the correct task head).
            loader:  Test DataLoader for this task.

        Returns:
            Top-1 accuracy in [0, 1].
        """
        raise NotImplementedError
