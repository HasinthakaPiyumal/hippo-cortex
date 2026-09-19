"""
Tests for NullSpaceProjector.

OWNER: Praveen Dedigama
"""
import pytest
import torch
from hippocortex.cl.null_space_projector import NullSpaceProjector


def test_projection_removes_basis_component():
    """
    If old hidden states lie entirely along the first feature direction,
    then that direction should become protected.

    A gradient lying entirely in that protected direction
    should be removed by the projector.
    """
    projector = NullSpaceProjector(rank_budget=2)

    hidden_states = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )

    projector.update(hidden_states)

    grad = torch.tensor([[1.0, 0.0]])

    projected = projector.project(grad)

    expected = torch.tensor([[0.0, 0.0]])

    assert torch.allclose(projected, expected, atol=1e-6)


def test_projection_identity_before_any_task():
    """
    Before any task has been stored, there is no protected subspace.

    Therefore project(grad) should return the original gradient unchanged.
    """
    projector = NullSpaceProjector(rank_budget=2)

    grad = torch.tensor([[3.0, 4.0]])

    projected = projector.project(grad)

    assert torch.allclose(projected, grad, atol=1e-6)


def test_rank_budget_respected():
    """
    The number of stored protected directions must never exceed rank_budget.
    """
    projector = NullSpaceProjector(rank_budget=2)

    hidden_states = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    projector.update(hidden_states)
    print(projector._U)
    print("rank =", projector.current_rank)

    assert projector.current_rank <= 2

def test_projected_gradient_is_orthogonal_to_protected_basis():
    """
    After projection, the gradient should have approximately zero
    component along every protected direction.
    """
    projector = NullSpaceProjector(rank_budget=2)

    hidden_states = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
    )

    projector.update(hidden_states)

    grad = torch.tensor([[3.0, 4.0]])

    projected = projector.project(grad)

    protected_component = projected @ projector._U

    expected = torch.zeros_like(protected_component)

    assert torch.allclose(protected_component, expected, atol=1e-6)


def test_zero_hidden_states_do_not_create_protected_directions():
    """
    If hidden states contain no feature information, the projector
    should not create any protected directions.
    """
    projector = NullSpaceProjector(rank_budget=2)

    hidden_states = torch.zeros(3, 2)

    projector.update(hidden_states)

    assert projector.current_rank == 0