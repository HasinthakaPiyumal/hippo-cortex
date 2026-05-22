"""
SWRGenerator — Conditional VAE that simulates sharp-wave-ripple replay.

OWNER: Praveen Dedigama

Implementation notes:
- Architecture: Encoder MLP(d_model + task_emb_dim → 512 → 512 → [µ, log_var])
                Decoder MLP(latent_dim + task_emb_dim → 512 → 512 → d_model)
- Task embedding: nn.Embedding(n_tasks, 32), concatenated to input.
- Loss: ELBO = MSE reconstruction + KL divergence (β = 1.0).
- sample() is called during the sleep phase: given stored (µ_stored, σ²_stored)
  from StatsBuffer, draw z ~ N(µ_stored, σ²_stored) and decode to Ĥ_past.

References:
  Sohn et al. 2015 — CVAE (papers/stage1-swr-replay/)
  Shin et al. 2017 — Deep Generative Replay (papers/stage1-swr-replay/)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SWRGenerator(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int = 64, n_tasks: int = 20):
        """
        Args:
            hidden_dim:  Dimensionality of Mamba hidden states (= d_model of backbone).
            latent_dim:  Dimensionality of CVAE latent space z.
            n_tasks:     Total number of tasks (for task embedding table size).
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, h: Tensor, task_id: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        ELBO forward pass for training the generator on real hidden states.

        Args:
            h:       Real hidden states from backbone, shape (B, hidden_dim).
            task_id: Integer task index for conditioning.

        Returns:
            recon_h:  Reconstructed hidden states, shape (B, hidden_dim).
            mu_q:     Posterior mean,     shape (B, latent_dim).
            logvar_q: Posterior log-var,  shape (B, latent_dim).
        """
        raise NotImplementedError

    def sample(
        self,
        task_id: int,
        n_samples: int,
        stats: tuple[Tensor, Tensor],
        device: torch.device | None = None,
    ) -> Tensor:
        """
        Sample synthetic hidden states given stored task statistics.

        Args:
            task_id:   Task to replay.
            n_samples: Number of synthetic samples to generate.
            stats:     (mu, sigma2) tensors from StatsBuffer, each shape (hidden_dim,).
            device:    Target device; defaults to the generator's parameter device.

        Returns:
            Synthetic hidden states Ĥ_past, shape (n_samples, hidden_dim).
        """
        raise NotImplementedError

    @staticmethod
    def elbo_loss(
        recon_h: Tensor,
        h: Tensor,
        mu_q: Tensor,
        logvar_q: Tensor,
        beta: float = 1.0,
    ) -> Tensor:
        """
        Compute ELBO = MSE reconstruction + β * KL divergence.

        Returns a scalar loss tensor.
        """
        raise NotImplementedError
