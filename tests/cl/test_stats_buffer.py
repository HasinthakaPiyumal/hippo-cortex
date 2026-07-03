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
    buf = StatsBuffer()
    H = torch.randn(10, 128)
    buf.update(0, H)
    mu, var = buf.get_stats(0)
    assert mu.shape == (128,)
    assert var.shape == (128,)
    assert torch.allclose(mu, H.mean(dim=0))
    assert torch.allclose(var, H.var(dim=0, unbiased=False))


def test_stats_buffer_memory_is_linear():
    """
    memory_bytes() must grow linearly with number of tasks, not with
    number of training samples. This is the constant-memory invariant.
    """
    buf = StatsBuffer()
    H1 = torch.randn(10, 128)
    buf.update(0, H1)
    mem_1 = buf.memory_bytes()

    # Update with more samples from the same task
    H2 = torch.randn(100, 128)
    buf.update(0, H2)
    mem_2 = buf.memory_bytes()
    assert mem_1 == mem_2  # Memory footprint must not grow with samples

    # Add a new task
    H3 = torch.randn(50, 128)
    buf.update(1, H3)
    mem_3 = buf.memory_bytes()
    assert mem_3 == 2 * mem_1  # Memory footprint grows linearly with task count


def test_stats_buffer_memory_formula():
    """memory_bytes() == n_tasks * d_model * 2 * element_size_bytes."""
    buf = StatsBuffer()
    H = torch.randn(5, 64)
    buf.update(0, H)
    expected_bytes = 1 * 64 * 2 * 4  # 1 task, d_model=64, 2 tensors, float32=4 bytes
    assert buf.memory_bytes() == expected_bytes


def test_stats_buffer_missing_task_raises():
    """get_stats() for an unknown task_id must raise KeyError."""
    buf = StatsBuffer()
    with pytest.raises(KeyError):
        buf.get_stats(0)

