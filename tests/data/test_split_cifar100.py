"""
Tests for Split-CIFAR100 data loader.

OWNER: Induwara Mihisara

Integration tests: they run only when CIFAR-100 is extracted under data/cifar100
(or the directory named by $HIPPOCORTEX_CIFAR100_ROOT) and are skipped otherwise.
Download with: python scripts/download_datasets.py --dataset cifar100
"""
import os
from pathlib import Path

import pytest

from hippocortex.data.split_cifar100 import (
    N_CLASSES_PER_TASK,
    N_TASKS,
    get_task_classes,
    get_task_loaders,
    is_available,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
CIFAR_ROOT = Path(os.environ.get("HIPPOCORTEX_CIFAR100_ROOT", _REPO_ROOT / "data" / "cifar100"))

requires_cifar100 = pytest.mark.skipif(
    not is_available(CIFAR_ROOT),
    reason=f"requires CIFAR-100 under {CIFAR_ROOT} — run scripts/download_datasets.py",
)


@requires_cifar100
def test_task_loader_output_shape():
    """Loader must yield images (B, 3, 32, 32) and labels in [0, N_CLASSES_PER_TASK)."""
    loader = get_task_loaders(0, "train", CIFAR_ROOT, batch_size=16, num_workers=0)
    images, labels = next(iter(loader))
    assert images.shape == (16, 3, 32, 32)
    assert labels.min() >= 0 and labels.max() < N_CLASSES_PER_TASK

    # Every remapped label occurs in the task, and the task has 500 train images per class.
    train_labels = {loader.dataset[i][1] for i in range(0, len(loader.dataset), 50)}
    assert train_labels == set(range(N_CLASSES_PER_TASK))
    assert len(loader.dataset) == 500 * N_CLASSES_PER_TASK


@requires_cifar100
def test_no_label_overlap_between_tasks():
    """Label sets of different tasks must be disjoint."""
    task_classes = get_task_classes(seed=42)
    seen: set[int] = set()
    for t in range(N_TASKS):
        classes = set(task_classes[t])
        assert len(classes) == N_CLASSES_PER_TASK
        assert seen.isdisjoint(classes)
        seen |= classes
    assert seen == set(range(100))

    # The loaders draw their samples from the task's own original classes.
    for t in (0, 1):
        ds = get_task_loaders(t, "test", CIFAR_ROOT, num_workers=0).dataset
        originals = {ds.base.targets[i] for i in ds.indices}
        assert originals == set(task_classes[t])


@requires_cifar100
def test_all_20_tasks_loadable():
    """All 20 task loaders must construct without error."""
    total = 0
    for t in range(N_TASKS):
        loader = get_task_loaders(t, "test", CIFAR_ROOT, batch_size=8, num_workers=0)
        assert len(loader.dataset) == 100 * N_CLASSES_PER_TASK
        total += len(loader.dataset)
    assert total == 10_000
