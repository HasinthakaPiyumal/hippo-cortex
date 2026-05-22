"""
Tests for MambaBackbone.

OWNER: Praveen Dedigama
"""
import pytest
import torch
from hippocortex.models.backbone import MambaBackbone


def test_forward_output_shapes():
    """forward() must return logits (B, n_classes) and hidden (B, d_model)."""
    pytest.skip("implement when MambaBackbone lands")


def test_extract_hidden_shape():
    """extract_hidden() must return (B, d_model)."""
    pytest.skip("implement when MambaBackbone lands")


def test_set_task_head_changes_output_dim():
    """After set_task_head(n), forward returns logits of shape (B, n)."""
    pytest.skip("implement when MambaBackbone lands")
