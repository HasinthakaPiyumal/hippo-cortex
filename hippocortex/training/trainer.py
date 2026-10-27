"""
Trainer — orchestrates the 3-phase continual learning loop.

OWNER: Hasinthaka Piyumal

Three phases per task:
  1. Warmup:        Supervised CE on current task only.
                    No replay, no projection. ← IMPLEMENTED
  2. Joint:         CE on current task + replay term on SWR-generated past.
                    Gradients are projected through NullSpaceProjector. ← IMPLEMENTED
  3. Consolidation: Call consolidate() to update StatsBuffer and NullSpaceProjector.
                    TODO(Hasinthaka): wire in after stats_buffer + consolidation ready.

evaluate() fills one row of the acc_matrix after each task completes.

Phase 2 notes
-------------
* Projection scope. NullSpaceProjector holds one basis U of shape (d_model, k) built
  from the backbone's pre-logits features, so it can only project a gradient that has a
  d_model-sized axis: the task head (…, d_model) and out_proj.weight (d_model, d_inner).
  x_proj_* (…, d_inner) and A_log (d_inner, d_state) have no such axis and are SKIPPED
  (one-time warning; counts are returned in the metrics). Until per-parameter bases exist
  (as in Mamba-CL's ModAdam) those parameters train unprotected.
* Projection point. The gradient is projected before AdamW, which rescales every element
  and can move the update out of the null space. Mamba-CL projects the update instead.
* Replay term. Past hidden states come from the SWR generator, i.e. they live in the
  output space of the backbone, so no replay loss on them can send gradient into the
  backbone. _replay_loss() is the single place to change that; the default is the
  ELBO of the TODO, computed without gradients (a monitored value, not a training signal).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
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

        # fp16 autocast + GradScaler (as Mamba-CL --use_amp); the T4 has no usable bf16.
        self.use_amp = self.device.type == "cuda" and bool(cfg.training.get("use_amp", True))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Rebuilt for every task (see _build_optimizer). swr_gen is trained inside
        # consolidate() during the sleep phase and is not part of it.
        self.optimizer: torch.optim.Optimizer | None = None
        self._criterion = nn.CrossEntropyLoss()
        self._warned_skipped_projection = False

        _log.info(
            "Trainer initialised | device=%s | amp=%s | lr=%s | epochs_warmup=%d | epochs_joint=%d",
            self.device,
            self.use_amp,
            cfg.training.lr,
            cfg.training.epochs_warmup,
            cfg.training.epochs_joint,
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
            "loss_consolidation", "replay_loss", plus "proj_params_projected" and
            "proj_params_skipped" (parameters whose gradient was / was not projected).
        """
        self._start_task(task_id)

        # ── Phase 1: Warmup ────────────────────────────────────────────
        # Supervised cross-entropy on the current task. No replay, no projection.
        warmup_loss = self._run_warmup(task_id, loader)
        _log.info("Task %d | warmup done | avg_loss=%.4f", task_id, warmup_loss)

        # ── Phase 2: Joint ─────────────────────────────────────────────
        # CE on the current task + replay term, gradients projected, clipped, stepped.
        # Task 0 (or any task with an empty buffer) has no replay and, until the first
        # consolidation, no projection basis either.
        joint = self._run_joint(task_id, loader)
        _log.info(
            "Task %d | joint done | avg_loss=%.4f | replay=%.4f | projected=%d skipped=%d",
            task_id,
            joint["loss_joint"],
            joint["replay_loss"],
            joint["proj_params_projected"],
            joint["proj_params_skipped"],
        )

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

        return {
            "task_id": float(task_id),
            "loss_warmup": warmup_loss,
            "loss_joint": joint["loss_joint"],
            "loss_consolidation": consolidation_loss,
            "replay_loss": joint["replay_loss"],
            "proj_params_projected": joint["proj_params_projected"],
            "proj_params_skipped": joint["proj_params_skipped"],
        }

    def evaluate(self, task_id: int, loader: DataLoader) -> float:
        """
        Evaluate the backbone on a single task's loader, using that task's own head.

        Args:
            task_id: Task to evaluate; selects head ``task_id`` of the backbone.
            loader:  Eval DataLoader for this task.
                     Labels must be remapped to [0, n_classes_per_task).

        Returns:
            Top-1 accuracy in [0, 1].
        """
        was_training = self.backbone.training
        self.backbone.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                with self._autocast():
                    logits, _hidden = self.backbone(images, task_id=task_id)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.backbone.train(was_training)
        acc = correct / max(total, 1)
        _log.info("Task %d | top-1 acc=%.4f (%d/%d)", task_id, acc, correct, total)
        return acc

    # ------------------------------------------------------------------
    # Task setup
    # ------------------------------------------------------------------

    def _autocast(self) -> torch.autocast:
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp)

    def _start_task(self, task_id: int) -> None:
        """Add the task's head, then build an optimizer that contains it."""
        self.backbone.set_task_head(self.cfg.dataset.n_classes_per_task, task_id=task_id)
        self.backbone.to(self.device)
        self.optimizer = self._build_optimizer(task_id)
        self.backbone.train()

    def _build_optimizer(self, task_id: int) -> torch.optim.Optimizer:
        """
        Fresh AdamW per task, as Mamba-CL does: trainable backbone parameters plus the
        CURRENT task's head only (earlier heads are frozen). Building it after
        set_task_head() is what puts the new head's parameters in the optimizer.
        """
        t = self.cfg.training
        lr = float(t.lr)
        groups = [
            {"params": self.backbone.backbone_parameters(), "lr": lr},
            {
                "params": self.backbone.head_parameters(task_id),
                "lr": lr * float(t.get("head_lr_scale", 1.0)),
            },
        ]
        return torch.optim.AdamW(
            groups,
            lr=lr,
            betas=tuple(t.get("betas", (0.9, 0.999))),
            weight_decay=float(t.get("weight_decay", 0.01)),
        )

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def _run_warmup(self, task_id: int, loader: DataLoader) -> float:
        epochs = int(self.cfg.training.epochs_warmup)
        loss_sum = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    logits, _hidden = self.backbone(images, task_id=task_id)
                    loss = self._criterion(logits.float(), labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                epoch_steps += 1

            loss_sum += epoch_loss / max(epoch_steps, 1)
            _log.debug(
                "Task %d | warmup epoch %d/%d | loss=%.4f",
                task_id,
                epoch + 1,
                epochs,
                epoch_loss / max(epoch_steps, 1),
            )

        return loss_sum / max(epochs, 1)

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def _run_joint(self, task_id: int, loader: DataLoader) -> dict[str, float]:
        epochs = int(self.cfg.training.epochs_joint)
        loss_sum = 0.0
        replay_sum = 0.0
        projected = 0
        skipped = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_replay = 0.0
            epoch_steps = 0

            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                total, replay, projected, skipped = self._joint_step(task_id, images, labels)
                epoch_loss += total
                epoch_replay += replay
                epoch_steps += 1

            loss_sum += epoch_loss / max(epoch_steps, 1)
            replay_sum += epoch_replay / max(epoch_steps, 1)
            _log.debug(
                "Task %d | joint epoch %d/%d | loss=%.4f | replay=%.4f",
                task_id,
                epoch + 1,
                epochs,
                epoch_loss / max(epoch_steps, 1),
                epoch_replay / max(epoch_steps, 1),
            )

        return {
            "loss_joint": loss_sum / max(epochs, 1),
            "replay_loss": replay_sum / max(epochs, 1),
            "proj_params_projected": float(projected),
            "proj_params_skipped": float(skipped),
        }

    def _joint_step(
        self, task_id: int, images: Tensor, labels: Tensor
    ) -> tuple[float, float, int, int]:
        """
        One Phase-2 batch: CE + replay -> backward -> unscale -> project -> clip -> step.

        Returns:
            (total_loss, replay_loss, n_params_projected, n_params_skipped)
        """
        self.optimizer.zero_grad(set_to_none=True)

        with self._autocast():
            logits, _hidden = self.backbone(images, task_id=task_id)
            ce_loss = self._criterion(logits.float(), labels)

        replay_loss = self._replay_loss(task_id, n_samples=images.shape[0])
        total_loss = ce_loss + replay_loss

        self.scaler.scale(total_loss).backward()
        # Gradients must be in true scale before they are projected and clipped.
        self.scaler.unscale_(self.optimizer)

        n_projected, n_skipped = self._project_gradients()

        grad_clip = self.cfg.training.get("grad_clip")
        if grad_clip is not None:
            params = [p for p in self.backbone.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, float(grad_clip))

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss.item(), replay_loss.item(), n_projected, n_skipped

    def _replay_loss(self, task_id: int, n_samples: int) -> Tensor:
        """
        Replay term for the current batch; zero when no earlier task is in the buffer.

        Default: sample ``n_samples`` hidden states per earlier task from the SWR
        generator and return the mean ELBO (kl_weight = beta), as in the original TODO.
        It runs under no_grad: the sampled states do not depend on the backbone, so the
        term carries no gradient, and the generator (trained in consolidate()) must not
        collect stray gradients here. Replace this method to change what replay trains.
        """
        past = [t for t in self.buffer.task_ids() if t < task_id]
        if not past:
            return torch.zeros((), device=self.device)

        beta = float(self.cfg.training.get("kl_weight", 1.0))
        losses = []
        with torch.no_grad():
            for t in past:
                h_past = self.swr_gen.sample(t, n_samples, self.buffer.get_stats(t), self.device)
                recon, mu_q, logvar_q = self.swr_gen(h_past, t)
                losses.append(SWRGenerator.elbo_loss(recon, h_past, mu_q, logvar_q, beta=beta))
        return torch.stack(losses).mean()

    def _project_gradients(self) -> tuple[int, int]:
        """
        Project p.grad through the null-space projector wherever its shape allows.

        NullSpaceProjector.project() acts on the LAST axis of its input and raises if that
        axis is not d_model, so per parameter:
          * grad.shape[-1] == d_model            -> project(grad)            (e.g. head weight)
          * 2-D grad with grad.shape[0] == d_model -> project(grad.T).T      (e.g. out_proj.weight)
          * otherwise                             -> skipped, left unprojected.
        With no basis yet (before the first consolidation) nothing is touched.

        Returns:
            (n_projected, n_skipped)
        """
        d_model = self.projector.feature_dim
        if d_model is None:
            return 0, 0

        projected = 0
        skipped: list[str] = []
        for name, p in self.backbone.named_parameters():
            g = p.grad
            if g is None or not p.requires_grad:
                continue
            if g.shape[-1] == d_model:
                p.grad = self.projector.project(g)
            elif g.ndim == 2 and g.shape[0] == d_model:
                p.grad = self.projector.project(g.T).T.contiguous()
            else:
                skipped.append(name)
                continue
            projected += 1

        if skipped and not self._warned_skipped_projection:
            self._warned_skipped_projection = True
            _log.warning(
                "Null-space projection skipped for %d parameters with no axis of size %d "
                "(they train unprojected), e.g. %s. This warning is shown once.",
                len(skipped),
                d_model,
                skipped[:3],
            )
        return projected, len(skipped)
