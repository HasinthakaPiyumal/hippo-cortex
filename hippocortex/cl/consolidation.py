"""
Consolidation driver — the SWR sleep phase (Phase 3).

OWNER: Hasinthaka Piyumal

Called once per task, after the awake phases (warmup + joint). The backbone is NOT updated
here; it was protected during Phase 2 by the null-space projector. Sleep does three things:

  1. Replay: draw n_samples synthetic hidden states for every earlier task from the SWR
     generator *as it was before this sleep* (its stored (µ, σ²) plus the CVAE). These are
     the generator's own memory of the old tasks (deep generative replay, Shin et al. 2017).
  2. Fine-tune the CVAE on the current task's real hidden states together with the replayed
     ones, so it learns the new task without forgetting how to generate the old ones.
  3. Bookkeeping with the current task's real hidden states:
       StatsBuffer.update(task_id, H)   -> (µ, σ²) used by sample() from now on
       NullSpaceProjector.update(H)     -> extends the protected basis U for later tasks

Implementation notes:
- The generator's gradients are not projected: U lives in the backbone's feature space, and
  replaying the generator's own samples is what protects it.
- Replayed states are sampled once, before fine-tuning, so the generator never trains on its
  own drifting output.
- Each step averages the ELBO over the current task and every replayed task, so all tasks
  weigh the same however many there are.
- Stored statistics are not refreshed when the backbone drifts on later tasks; the projector
  is what keeps that drift small.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hippocortex.models.backbone import MambaBackbone

from hippocortex.cl.null_space_projector import NullSpaceProjector
from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.models.swr_generator import SWRGenerator


def collect_hidden_states(
    backbone: MambaBackbone,
    loader: DataLoader,
    device: torch.device,
    autocast: Callable[[], object] | None = None,
) -> Tensor:
    """
    Hidden states of every sample in ``loader``: backbone.extract_hidden in eval mode,
    without gradients. The backbone's train/eval mode is restored afterwards.

    Args:
        backbone: The backbone (extract_hidden(images) -> (B, d_model)).
        loader:   Yields (images, labels); labels are ignored.
        device:   Device the images are moved to.
        autocast: Optional context factory (e.g. the trainer's fp16 autocast).

    Returns:
        (N, d_model) float32 tensor on ``device``.
    """
    was_training = backbone.training
    backbone.eval()
    chunks = []
    with torch.no_grad():
        for images, _labels in loader:
            with (autocast() if autocast is not None else nullcontext()):
                h = backbone.extract_hidden(images.to(device))
            chunks.append(h.float())
    backbone.train(was_training)
    if not chunks:
        raise ValueError("collect_hidden_states: the loader is empty")
    return torch.cat(chunks)


def consolidate(
    swr_gen: SWRGenerator,
    projector: NullSpaceProjector,
    buffer: StatsBuffer,
    task_id: int,
    optimizer: Optimizer,
    current_hidden_states: Tensor,
    n_samples: int = 512,
    epochs: int = 3,
    batch_size: int = 256,
    beta: float = 1.0,
) -> float:
    """
    Run the sleep/SWR consolidation phase for the task that just completed.

    Args:
        swr_gen:               The conditional VAE SWR generator (fine-tuned here).
        projector:             The null-space gradient projector (basis extended here).
        buffer:                The per-task stats buffer (task_id stored here).
        task_id:               Index of the task that just finished awake training.
        optimizer:             Optimizer over swr_gen's parameters.
        current_hidden_states: Real hidden states of task_id, shape (N, d_model).
        n_samples:             Synthetic samples drawn per earlier task.
        epochs:                Passes over the current task's hidden states. 0 skips the
                               CVAE fine-tune and only does the bookkeeping.
        batch_size:            Current-task states per step (the same number is drawn from
                               each replayed task).
        beta:                  KL weight of the ELBO.

    Returns:
        Mean ELBO over all fine-tuning steps (0.0 when epochs == 0), for logging.
    """
    h_cur = current_hidden_states.detach().float()
    if h_cur.ndim != 2 or h_cur.shape[1] != swr_gen.hidden_dim:
        raise ValueError(
            f"current_hidden_states must be (N, {swr_gen.hidden_dim}), got {tuple(h_cur.shape)}"
        )
    device = next(swr_gen.parameters()).device
    h_cur = h_cur.to(device)

    # 1. Replay targets from the generator as it is now, before it learns the new task.
    past = [t for t in buffer.task_ids() if t != task_id]
    with torch.no_grad():
        replay = {t: swr_gen.sample(t, n_samples, buffer.get_stats(t), device).detach() for t in past}

    # 2. Fine-tune the CVAE on current real states + replayed old states.
    losses = []
    swr_gen.train()
    for _epoch in range(epochs):
        order = torch.randperm(h_cur.shape[0], device=device)
        for start in range(0, h_cur.shape[0], batch_size):
            batch = h_cur[order[start:start + batch_size]]
            recon, mu_q, logvar_q = swr_gen(batch, task_id)
            terms = [SWRGenerator.elbo_loss(recon, batch, mu_q, logvar_q, beta=beta)]
            for t, h_old in replay.items():
                idx = torch.randint(0, h_old.shape[0], (batch.shape[0],), device=device)
                old = h_old[idx]
                recon, mu_q, logvar_q = swr_gen(old, t)
                terms.append(SWRGenerator.elbo_loss(recon, old, mu_q, logvar_q, beta=beta))
            loss = torch.stack(terms).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    # 3. Bookkeeping with the real states of the task that just finished.
    buffer.update(task_id, h_cur)
    projector.update(h_cur)

    return sum(losses) / len(losses) if losses else 0.0
