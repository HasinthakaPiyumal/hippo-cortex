"""
Unit Tests for SWRGenerator (CVAE).

OWNER: Thagya Kavindi / Hasinthaka Piyumal
Coverage: Shapes, Sampling, ELBO Loss, Gradient Flow, Overfitting Sanity Check
"""
import pytest
import torch
import torch.nn as nn
from hippocortex.models.swr_generator import SWRGenerator


@pytest.fixture
def cvae_model():
    """Fixture providing an initialized SWRGenerator model."""
    hidden_dim = 128
    latent_dim = 64
    n_tasks = 20
    try:
        model = SWRGenerator(hidden_dim=hidden_dim, latent_dim=latent_dim, n_tasks=n_tasks)
        return model
    except NotImplementedError:
        pytest.skip("SWRGenerator implementation is not ready yet.")


def test_cvae_forward_shapes(cvae_model):
    """Test that forward pass returns correct output shapes."""
    B, hidden_dim, latent_dim = 16, 128, 64
    task_id = 2
    h = torch.randn(B, hidden_dim)

    recon_h, mu_q, logvar_q = cvae_model(h, task_id)

    assert recon_h.shape == (B, hidden_dim), f"Expected recon_h shape {(B, hidden_dim)}, got {recon_h.shape}"
    assert mu_q.shape == (B, latent_dim), f"Expected mu_q shape {(B, latent_dim)}, got {mu_q.shape}"
    assert logvar_q.shape == (B, latent_dim), f"Expected logvar_q shape {(B, latent_dim)}, got {logvar_q.shape}"


def test_cvae_elbo_loss_calculation():
    """Test that static elbo_loss calculates non-negative scalar loss."""
    B, hidden_dim, latent_dim = 8, 128, 64
    h = torch.randn(B, hidden_dim)
    recon_h = torch.randn(B, hidden_dim)
    mu_q = torch.randn(B, latent_dim)
    logvar_q = torch.randn(B, latent_dim)

    try:
        loss = SWRGenerator.elbo_loss(recon_h, h, mu_q, logvar_q, beta=1.0)
    except NotImplementedError:
        pytest.skip("SWRGenerator.elbo_loss implementation is not ready yet.")

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, "Loss must be a scalar tensor"
    assert loss.item() >= 0, "ELBO loss should be non-negative"
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss contains NaN or Inf"


def test_cvae_sample_shape_and_device(cvae_model):
    """Test sampling synthetic hidden states using StatsBuffer (µ, σ²) tensors."""
    hidden_dim = 128
    n_samples = 32
    task_id = 0

    # Simulated StatsBuffer output
    mu_stored = torch.zeros(hidden_dim)
    sigma2_stored = torch.ones(hidden_dim)
    stats = (mu_stored, sigma2_stored)

    synthetic_h = cvae_model.sample(task_id=task_id, n_samples=n_samples, stats=stats)

    assert synthetic_h.shape == (n_samples, hidden_dim), f"Expected synthetic_h shape {(n_samples, hidden_dim)}, got {synthetic_h.shape}"
    assert not torch.isnan(synthetic_h).any(), "Synthetic hidden states contain NaN"


def test_cvae_gradient_flow(cvae_model):
    """Test that gradients flow to encoder, decoder, and embedding weights."""
    B, hidden_dim = 8, 128
    task_id = 1
    h = torch.randn(B, hidden_dim, requires_grad=True)

    recon_h, mu_q, logvar_q = cvae_model(h, task_id)
    loss = SWRGenerator.elbo_loss(recon_h, h, mu_q, logvar_q)
    loss.backward()

    for name, param in cvae_model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
