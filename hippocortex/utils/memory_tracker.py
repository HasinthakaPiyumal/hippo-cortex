"""Utilities for measuring memory footprint of model components."""
import sys
import torch
import torch.nn as nn


def tensor_bytes(t: torch.Tensor) -> int:
    """Return byte size of a tensor."""
    return t.nelement() * t.element_size()


def measure_memory_mb(obj: object) -> float:
    """
    Approximate memory usage of an object in MB.
    Walks tensor attributes recursively; does not account for Python overhead.
    """
    raise NotImplementedError


def model_parameter_mb(model: nn.Module) -> float:
    """Return total parameter memory of an nn.Module in MB."""
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    return total / (1024 ** 2)
