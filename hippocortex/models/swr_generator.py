"""
SWRGenerator — Conditional VAE that simulates sharp-wave-ripple replay.

OWNER: Praveen Dedigama

Implementation notes:
- Architecture:
    Encoder MLP(d_model + task_emb_dim -> 512 -> 512 -> [mu, log_var])
    Decoder MLP(latent_dim + task_emb_dim -> 512 -> 512 -> d_model)

- Task embedding:
    nn.Embedding(n_tasks, 32), concatenated to encoder and decoder inputs.

- Loss:
    ELBO = MSE reconstruction + beta * KL divergence.

- sample():
    Given stored hidden-state statistics (mu, sigma^2) from StatsBuffer,
    first draw rough hidden states in hidden-state space, then pass them
    through the conditional VAE to produce synthetic replay hidden states.

References:
  Sohn et al. 2015 — CVAE
  Shin et al. 2017 — Deep Generative Replay
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SWRGenerator(nn.Module):
    
    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = 64,
        n_tasks: int = 20,
    ):
        """
        Args:
            hidden_dim:
                Dimensionality of Mamba hidden states
                (= d_model of backbone).

            latent_dim:
                Dimensionality of CVAE latent space z.

            n_tasks:
                Total number of tasks
                (for task embedding table size).
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_tasks = n_tasks

        task_emb_dim = 32
        self.task_emb_dim = task_emb_dim

        # Each task gets a learnable embedding vector.
        self.task_embedding = nn.Embedding(
            n_tasks,
            task_emb_dim,
        )

        # Encoder input:
        # hidden state + task embedding
        #
        # hidden_dim + task_emb_dim
        # Example:
        # 128 + 32 = 160
        self.encoder = nn.Sequential(
            nn.Linear(
                hidden_dim + task_emb_dim,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                512,
            ),
            nn.ReLU(),
        )

        # Posterior mean in latent space.
        self.mu_layer = nn.Linear(
            512,
            latent_dim,
        )

        # Posterior log-variance in latent space.
        self.logvar_layer = nn.Linear(
            512,
            latent_dim,
        )

        # Decoder input:
        # latent vector z + task embedding
        #
        # latent_dim + task_emb_dim
        # Example:
        # 64 + 32 = 96
        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dim + task_emb_dim,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                512,
            ),
            nn.ReLU(),
            nn.Linear(
                512,
                hidden_dim,
            ),
        )

    def forward(
        self,
        h: Tensor,
        task_id: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        ELBO forward pass for training the generator on hidden states.

        Args:
            h:
                Hidden states from backbone,
                shape (B, hidden_dim).

            task_id:
                Integer task index used for conditioning.

        Returns:
            recon_h:
                Reconstructed hidden states,
                shape (B, hidden_dim).

            mu_q:
                Posterior mean,
                shape (B, latent_dim).

            logvar_q:
                Posterior log-variance,
                shape (B, latent_dim).
        """

        # Number of hidden states in the current batch.
        batch_size = h.shape[0]

        # Create one task id for each sample in the batch.
        #
        # Example:
        # batch_size = 4
        # task_id = 2
        #
        # task_ids = [2, 2, 2, 2]
        task_ids = torch.full(
            (batch_size,),
            task_id,
            dtype=torch.long,
            device=h.device,
        )

        # Convert task ids into learned task embeddings.
        #
        # Shape:
        # (B, task_emb_dim)
        task_emb = self.task_embedding(
            task_ids
        )

        # Concatenate hidden state and task embedding.
        #
        # Example:
        # h        -> (B, 128)
        # task_emb -> (B, 32)
        #
        # output   -> (B, 160)
        encoder_input = torch.cat(
            [h, task_emb],
            dim=1,
        )

        # Encode into shared feature representation.
        #
        # Shape:
        # (B, 160) -> (B, 512)
        encoded = self.encoder(
            encoder_input
        )

        # Create latent distribution parameters.
        #
        # Both shapes:
        # (B, latent_dim)
        mu_q = self.mu_layer(
            encoded
        )

        logvar_q = self.logvar_layer(
            encoded
        )

        # Convert log-variance into standard deviation.
        #
        # variance = exp(logvar)
        # std = sqrt(variance)
        #     = exp(0.5 * logvar)
        std_q = torch.exp(
            0.5 * logvar_q
        )

        # Random normal noise with the same shape as std_q.
        epsilon = torch.randn_like(
            std_q
        )

        # Reparameterization trick.
        #
        # z = mu + std * epsilon
        z = (
            mu_q
            + std_q * epsilon
        )

        # Condition the decoder on the task as well.
        #
        # Example:
        # z        -> (B, 64)
        # task_emb -> (B, 32)
        #
        # output   -> (B, 96)
        decoder_input = torch.cat(
            [z, task_emb],
            dim=1,
        )

        # Decode back to hidden-state space.
        #
        # Shape:
        # (B, 96) -> (B, hidden_dim)
        recon_h = self.decoder(
            decoder_input
        )

        return recon_h, mu_q, logvar_q

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
            task_id:
                Task to replay.

            n_samples:
                Number of synthetic replay samples to generate.

            stats:
                (mu, sigma2) tensors from StatsBuffer.

                Each tensor has shape:
                (hidden_dim,)

            device:
                Target device.

                If None, use the generator's own device.

        Returns:
            Synthetic replay hidden states,
            shape (n_samples, hidden_dim).
        """

        # If device was not explicitly provided,
        # use the device of the generator itself.
        if device is None:
            device = next(
                self.parameters()
            ).device

        # Match the stored statistics to the generator dtype.
        generator_dtype = next(
            self.parameters()
        ).dtype

        # Unpack stored hidden-state statistics.
        mu_h, sigma2_h = stats

        # Move statistics to the same device and dtype
        # as the generator.
        mu_h = mu_h.to(
            device=device,
            dtype=generator_dtype,
        )

        sigma2_h = sigma2_h.to(
            device=device,
            dtype=generator_dtype,
        )

        # Variance should not be negative.
        # Clamp tiny numerical negatives to zero.
        sigma2_h = torch.clamp(
            sigma2_h,
            min=0.0,
        )

        # Convert variance to standard deviation.
        std_h = torch.sqrt(
            sigma2_h
        )

        # Generate random normal noise.
        #
        # Shape:
        # (n_samples, hidden_dim)
        epsilon = torch.randn(
            n_samples,
            self.hidden_dim,
            device=device,
            dtype=generator_dtype,
        )

        # Draw rough synthetic hidden states:
        #
        # h_rough = mu_h + std_h * epsilon
        #
        # mu_h and std_h have shape:
        # (hidden_dim,)
        #
        # unsqueeze(0) changes them to:
        # (1, hidden_dim)
        #
        # PyTorch then broadcasts them across all samples.
        rough_h = (
            mu_h.unsqueeze(0)
            + std_h.unsqueeze(0) * epsilon
        )

        # Refine the rough hidden states through the CVAE.
        recon_h, _, _ = self.forward(
            rough_h,
            task_id,
        )

        return recon_h

    @staticmethod
    def elbo_loss(
        recon_h: Tensor,
        h: Tensor,
        mu_q: Tensor,
        logvar_q: Tensor,
        beta: float = 1.0,
    ) -> Tensor:
        """
        Compute CVAE loss:

            loss =
                reconstruction_loss
                + beta * KL_loss

        Returns:
            Scalar loss tensor.
        """

        # Reconstruction loss:
        #
        # Measures how close reconstructed hidden states
        # are to the original hidden states.
        recon_loss = torch.mean(
            (recon_h - h) ** 2
        )

        # KL divergence:
        #
        # Keeps the learned latent distribution reasonably
        # close to a standard normal distribution.
        kl_loss = -0.5 * torch.mean(
            1
            + logvar_q
            - mu_q.pow(2)
            - logvar_q.exp()
        )

        # Final CVAE loss.
        loss = (
            recon_loss
            + beta * kl_loss
        )

        return loss