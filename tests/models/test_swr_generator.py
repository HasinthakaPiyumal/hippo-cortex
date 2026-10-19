"""
Tests for SWRGenerator.
"""

import torch

from hippocortex.models.swr_generator import SWRGenerator


def test_forward_output_shapes():
    """
    forward() should return tensors with the expected shapes.
    """
    generator = SWRGenerator(
        hidden_dim=128,
        latent_dim=64,
        n_tasks=20,
    )

    h = torch.randn(4, 128)

    recon_h, mu_q, logvar_q = generator(
        h,
        task_id=2,
    )

    assert recon_h.shape == (4, 128)
    assert mu_q.shape == (4, 64)
    assert logvar_q.shape == (4, 64)


def test_sample_output_shape():
    """
    sample() should generate the requested number of replay hidden states.
    """
    generator = SWRGenerator(
        hidden_dim=128,
        latent_dim=64,
        n_tasks=20,
    )

    mu_h = torch.zeros(128)
    sigma2_h = torch.ones(128)

    samples = generator.sample(
        task_id=1,
        n_samples=8,
        stats=(mu_h, sigma2_h),
    )

    assert samples.shape == (8, 128)


def test_elbo_loss_returns_scalar():
    """
    elbo_loss() should return a scalar tensor.
    """
    generator = SWRGenerator(
        hidden_dim=128,
        latent_dim=64,
        n_tasks=20,
    )

    h = torch.randn(4, 128)

    recon_h, mu_q, logvar_q = generator(
        h,
        task_id=0,
    )

    loss = SWRGenerator.elbo_loss(
        recon_h,
        h,
        mu_q,
        logvar_q,
    )

    assert loss.ndim == 0


def test_elbo_loss_is_non_negative():
    """
    ELBO-style loss should not be negative for normal inputs.
    """
    generator = SWRGenerator(
        hidden_dim=128,
        latent_dim=64,
        n_tasks=20,
    )

    h = torch.randn(4, 128)

    recon_h, mu_q, logvar_q = generator(
        h,
        task_id=0,
    )

    loss = SWRGenerator.elbo_loss(
        recon_h,
        h,
        mu_q,
        logvar_q,
    )

    assert loss.item() >= 0.0


def test_gradients_flow_through_generator():
    """
    Backpropagation should create gradients for generator parameters.
    """
    generator = SWRGenerator(
        hidden_dim=128,
        latent_dim=64,
        n_tasks=20,
    )

    h = torch.randn(4, 128)

    recon_h, mu_q, logvar_q = generator(
        h,
        task_id=3,
    )

    loss = SWRGenerator.elbo_loss(
        recon_h,
        h,
        mu_q,
        logvar_q,
    )

    loss.backward()

    grads = [
        p.grad
        for p in generator.parameters()
        if p.requires_grad
    ]

    assert any(
        grad is not None
        for grad in grads
    )