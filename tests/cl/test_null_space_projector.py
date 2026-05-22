"""
Tests for NullSpaceProjector.

OWNER: Praveen Dedigama
"""
import pytest
import torch
from hippocortex.cl.null_space_projector import NullSpaceProjector


def test_projection_removes_basis_component():
    """
    After one update with hidden states whose SVD basis is U[:,0],
    projecting U[:,0] must return (approximately) the zero vector.
    """
    pytest.skip("implement when NullSpaceProjector lands")


def test_projection_identity_before_any_task():
    """project(grad) must equal grad when no tasks have been seen yet."""
    pytest.skip("implement when NullSpaceProjector lands")


def test_rank_budget_respected():
    """current_rank must never exceed rank_budget."""
    pytest.skip("implement when NullSpaceProjector lands")
