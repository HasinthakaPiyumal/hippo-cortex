"""
Tests for StatsBuffer.

OWNER: Hasinthaka Piyumal

The core paper claim is constant memory footprint.
These tests must pass before any experiment results are reported.
"""
import pytest
import torch
from hippocortex.cl.stats_buffer import StatsBuffer


def test_stats_buffer_stores_and_retrieves():
    """update() followed by get_stats() must round-trip without error."""
    pytest.skip("implement when StatsBuffer lands")


def test_stats_buffer_memory_is_linear():
    """
    memory_bytes() must grow linearly with number of tasks, not with
    number of training samples. This is the constant-memory invariant.
    """
    pytest.skip("implement when StatsBuffer lands")


def test_stats_buffer_memory_formula():
    """memory_bytes() == n_tasks * d_model * 2 * element_size_bytes."""
    pytest.skip("implement when StatsBuffer lands")


def test_stats_buffer_missing_task_raises():
    """get_stats() for an unknown task_id must raise KeyError."""
    pytest.skip("implement when StatsBuffer lands")
