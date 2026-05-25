"""
Trainer — orchestrates the 3-phase continual learning loop.

OWNER: Hasinthaka Piyumal

Three phases per task:
  1. Warmup:        Supervised CE on current task only.
                    No replay, no projection. ← IMPLEMENTED
  2. Joint:         CE on current task + ELBO on SWR-generated past.
                    All backbone gradients projected through NullSpaceProjector.
                    TODO(Praveen): wire in after backbone + swr_gen are ready.
  3. Consolidation: Call consolidate() to update StatsBuffer and NullSpaceProjector.
                    TODO(Hasinthaka): wire in after stats_buffer + consolidation ready.

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
    """
    Continual learning trainer for HippoCortex Stage 1.

    Usage
    -----
    trainer = Trainer(backbone, swr_gen, projector, buffer, cfg)
    for task_id in range(cfg.dataset.n_tasks):
        metrics = trainer.train_task(task_id, train_loaders[task_id])
        acc = trainer.evaluate(task_id, val_loaders[task_id])
    """

    def __init__(
        self,
        backbone: MambaBackbone,
        swr_gen: SWRGenerator,
        projector: NullSpaceProjector,
        buffer: StatsBuffer,
        cfg: DictConfig,
    ) -> None:
        self.backbone = backbone
        self.swr_gen = swr_gen
        self.projector = projector
        self.buffer = buffer
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)
        self.swr_gen.to(self.device)

        # Optimizer covers backbone only.
        # swr_gen is trained inside consolidate() during the sleep phase.
        self.optimizer = torch.optim.AdamW(
            backbone.parameters(),
            lr=cfg.training.lr,
        )
        self._criterion = nn.CrossEntropyLoss()

        _log.info(
            "Trainer initialised | device=%s | lr=%s | epochs_warmup=%d",
            self.device,
            cfg.training.lr,
            cfg.training.epochs_warmup,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_task(self, task_id: int, loader: DataLoader) -> dict[str, float]:
        """
        Run all three phases for one task.

        Args:
            task_id: Integer task index (0-based).
            loader:  Training DataLoader for this task.
                     Labels must be remapped to [0, n_classes_per_task) by the loader.

        Returns:
            Dict with keys: "task_id", "loss_warmup", "loss_joint",
            "loss_consolidation", "replay_loss".
        """
        n_classes = self.cfg.dataset.n_classes_per_task

        # ── Phase 1: Warmup ────────────────────────────────────────────
        # Supervised cross-entropy on the current task. No replay, no projection.
        self.backbone.set_task_head(n_classes)
        self.backbone.to(self.device)
        self.backbone.train()

        warmup_loss_sum = 0.0
        warmup_steps = 0

        for epoch in range(self.cfg.training.epochs_warmup):
            epoch_loss = 0.0
            epoch_steps = 0

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits, _hidden = self.backbone(images)
                loss = self._criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

            warmup_loss_sum += epoch_loss / max(epoch_steps, 1)
            warmup_steps += 1
            _log.debug(
                "Task %d | warmup epoch %d/%d | loss=%.4f",
                task_id,
                epoch + 1,
                self.cfg.training.epochs_warmup,
                epoch_loss / max(epoch_steps, 1),
            )

        warmup_loss = warmup_loss_sum / max(warmup_steps, 1)
        _log.info("Task %d | warmup done | avg_loss=%.4f", task_id, warmup_loss)

        # ── Phase 2: Joint ─────────────────────────────────────────────
        # TODO(Praveen): Implement after MambaBackbone + SWRGenerator are ready.
        #
        # What goes here:
        #   - Loop cfg.training.epochs_joint epochs
        #   - Each batch:
        #       logits, hidden = backbone(images)
        #       ce_loss = criterion(logits, labels)
        #       h_past = swr_gen.sample(prior_task_id, n_samples, buffer.get_stats(...))
        #       recon, mu_q, logvar_q = swr_gen(h_past, prior_task_id)
        #       elbo = SWRGenerator.elbo_loss(recon, h_past, mu_q, logvar_q)
        #       total_loss = ce_loss + elbo
        #       total_loss.backward()
        #       for p in backbone.parameters():
        #           if p.grad is not None:
        #               p.grad = projector.project(p.grad)
        #       optimizer.step()
        joint_loss = 0.0

        # ── Phase 3: Consolidation ─────────────────────────────────────
        # TODO(Hasinthaka): Implement after StatsBuffer + NullSpaceProjector ready.
        #
        # What goes here:
        #   hidden_states = backbone.extract_hidden(all_task_images)
        #   replay_loss = consolidate(
        #       backbone, swr_gen, projector, buffer,
        #       task_id, optimizer, hidden_states,
        #       n_samples=512,
        #   )
        consolidation_loss = 0.0
        replay_loss = 0.0

        return {
            "task_id": float(task_id),
            "loss_warmup": warmup_loss,
            "loss_joint": joint_loss,
            "loss_consolidation": consolidation_loss,
            "replay_loss": replay_loss,
        }

    def evaluate(self, task_id: int, loader: DataLoader) -> float:
        """
        Evaluate the backbone on a single task's loader.

        Args:
            task_id: Task to evaluate.
                     Used only for logging; backbone head is already set for this task.
            loader:  Eval DataLoader for this task.
                     Labels must be remapped to [0, n_classes_per_task).

        Returns:
            Top-1 accuracy in [0, 1].
        """
        self.backbone.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits, _hidden = self.backbone(images)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / max(total, 1)
        _log.info("Task %d | top-1 acc=%.4f (%d/%d)", task_id, acc, correct, total)
        return acc
