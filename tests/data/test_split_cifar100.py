"""
Tests for Split-CIFAR100 data loader.

OWNER: Induwara Mihisara

Run only when data/cifar100 is available (mark as integration test).
"""
import pytest
from hippocortex.data.split_cifar100 import get_task_loaders, N_TASKS, N_CLASSES_PER_TASK


@pytest.mark.skip(reason="requires data/cifar100 — run manually after download")
def test_task_loader_output_shape(tmp_path):
    """Loader must yield images (B, 3, 32, 32) and labels in [0, N_CLASSES_PER_TASK)."""
    pass


@pytest.mark.skip(reason="requires data/cifar100 — run manually after download")
def test_no_label_overlap_between_tasks(tmp_path):
    """Label sets of different tasks must be disjoint."""
    pass


@pytest.mark.skip(reason="requires data/cifar100 — run manually after download")
def test_all_20_tasks_loadable(tmp_path):
    """All 20 task loaders must construct without error."""
    pass
