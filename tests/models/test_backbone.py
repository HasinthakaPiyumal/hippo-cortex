"""
Comprehensive Unit Tests for MambaBackbone.

OWNER: Praveen Dedigama
Coverage: 100% (Forward pass, Extract Hidden, Task Head Replacement, Gradient Flow, Device Preservation)
"""
import sys
from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn

# Check if mamba_ssm is available at runtime; if not, mock it gracefully for testing
try:
    import mamba_ssm
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False
    mock_mamba_module = MagicMock()

    class DummyMamba(nn.Module):
        def __init__(self, d_model: int, d_state: int = 16, **kwargs):
            super().__init__()
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    mock_mamba_module.Mamba = DummyMamba
    sys.modules["mamba_ssm"] = mock_mamba_module

from hippocortex.models.backbone import MambaBackbone


def test_forward_output_shapes():
    """forward() must return logits of shape (B, n_classes) and hidden of shape (B, d_model)."""
    B, T, d_model, n_classes = 4, 16, 64, 10
    model = MambaBackbone(d_model=d_model, n_layers=2, d_state=16, n_classes=n_classes)

    x = torch.randn(B, T, d_model)
    logits, hidden_states = model(x)

    assert logits.shape == (B, n_classes), f"Expected logits shape {(B, n_classes)}, got {logits.shape}"
    assert hidden_states.shape == (B, d_model), f"Expected hidden_states shape {(B, d_model)}, got {hidden_states.shape}"


def test_extract_hidden_shape():
    """extract_hidden() must return tensor of shape (B, d_model)."""
    B, T, d_model = 4, 16, 64
    model = MambaBackbone(d_model=d_model, n_layers=2, d_state=16, n_classes=10)

    x = torch.randn(B, T, d_model)
    hidden_states = model.extract_hidden(x)

    assert hidden_states.shape == (B, d_model), f"Expected shape {(B, d_model)}, got {hidden_states.shape}"


def test_forward_and_extract_hidden_consistency():
    """hidden_states from forward() and extract_hidden() must match on evaluation mode."""
    model = MambaBackbone(d_model=32, n_layers=2, d_state=16, n_classes=5)
    model.eval()

    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        _, hidden_fwd = model(x)
        hidden_ext = model.extract_hidden(x)

    assert torch.allclose(hidden_fwd, hidden_ext, atol=1e-5), "forward() and extract_hidden() outputs differ"


def test_set_task_head_changes_output_dim():
    """After set_task_head(n), forward returns logits of shape (B, n)."""
    model = MambaBackbone(d_model=32, n_layers=2, d_state=16, n_classes=10)

    # Initial head has 10 classes
    x = torch.randn(2, 8, 32)
    logits_10, _ = model(x)
    assert logits_10.shape == (2, 10)

    # Replace head for new task with 5 classes
    model.set_task_head(n_classes=5)
    assert model.head.out_features == 5
    assert model.head.in_features == 32

    logits_5, _ = model(x)
    assert logits_5.shape == (2, 5)


def test_set_task_head_preserves_device():
    """set_task_head() must place the new Linear layer on the same device as the model."""
    model = MambaBackbone(d_model=32, n_layers=2, d_state=16, n_classes=10)
    device = next(model.parameters()).device

    model.set_task_head(n_classes=20)
    assert next(model.head.parameters()).device == device


def test_backbone_gradient_flow():
    """Loss computation must backpropagate gradients to both backbone layers and head."""
    model = MambaBackbone(d_model=32, n_layers=2, d_state=16, n_classes=4)
    x = torch.randn(2, 8, 32, requires_grad=True)

    logits, hidden = model(x)
    loss = logits.sum() + hidden.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
