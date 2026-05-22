"""
Tests for continual learning metrics.

OWNER: Thagya Kavindi

Use a hand-crafted acc_matrix where AA/AF/BWT are known analytically.
Reference formulas: Lange et al. 2021, Appendix.
"""
import pytest
import numpy as np
from hippocortex.utils.metrics import average_accuracy, average_forgetting, backward_transfer, build_acc_matrix


def test_perfect_memory_no_forgetting():
    """
    If all diagonal entries equal 1.0 and no accuracy drops anywhere,
    AA = 1.0 and AF = 0.0.
    """
    pytest.skip("implement when metrics land")


def test_complete_forgetting():
    """
    If the model forgets everything after each task, AA = 1/T and AF is maximal.
    """
    pytest.skip("implement when metrics land")


def test_build_acc_matrix_initialised_negative():
    """build_acc_matrix must return a matrix with all values == -1."""
    m = build_acc_matrix(5)
    assert m.shape == (5, 5)
    assert (m == -1.0).all()
