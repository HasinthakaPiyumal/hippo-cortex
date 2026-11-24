"""
CPU tests for the augment option of the Split-CIFAR100 loader (no download).

OWNER: Hasinthaka Piyumal

The "inf_ssm" transforms are compared with baselines/inf-ssm/cl_datasets.py::build_transform
on the same image (and, for train, the same random state), and the loader is checked on a fake
CIFAR-100 that replaces torchvision's dataset class.
"""
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("timm")

from hippocortex.data import split_cifar100 as sc  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
INF_SSM_DIR = _REPO_ROOT / "baselines" / "inf-ssm"


def cifar_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))


def inf_ssm_args() -> SimpleNamespace:
    """The train_eval.py defaults that build_transform reads."""
    return SimpleNamespace(
        input_size=224, eval_crop_ratio=0.875, color_jitter=0.3, aa="rand-m9-mstd0.5-inc1",
        train_interpolation="bicubic", reprob=0.25, remode="pixel", recount=1, dataset="cifar100",
    )


@pytest.fixture(scope="module")
def cl_datasets():
    sys.path.insert(0, str(INF_SSM_DIR))
    try:
        import cl_datasets as mod
    finally:
        sys.path.remove(str(INF_SSM_DIR))
    return mod


def seeded(fn, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return fn()


# ── option parsing ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, mode", [(True, "cifar"), (False, "none"), (None, "none"),
                                         ("cifar", "cifar"), ("inf_ssm", "inf_ssm"), ("none", "none")])
def test_resolve_augment(value, mode):
    assert sc.resolve_augment(value) == mode


def test_unknown_augment_raises():
    with pytest.raises(ValueError, match="augment"):
        sc.resolve_augment("randaug")


# ── transforms ───────────────────────────────────────────────────────────────

def test_default_transforms_are_unchanged():
    img = cifar_image()
    eval_t = sc._build_transform(True, "cifar", train=False)
    assert eval_t(img).shape == (3, 32, 32)
    assert sc._build_transform(True, "cifar", train=True)(img).shape == (3, 32, 32)
    # True and "cifar" build the same train pipeline; eval has no random step.
    names = lambda t: [type(s).__name__ for s in t.transforms]  # noqa: E731
    assert names(sc._build_transform(True)) == names(sc._build_transform("cifar"))
    assert names(eval_t) == ["ToTensor", "Normalize"]
    assert names(sc._build_transform(False)) == ["ToTensor", "Normalize"]


@pytest.mark.parametrize("train", [True, False])
def test_inf_ssm_output_is_224(train):
    out = sc._build_transform("inf_ssm", train=train)(cifar_image())
    assert out.shape == (3, 224, 224)


def test_inf_ssm_eval_matches_the_baseline(cl_datasets):
    ours = sc._build_transform("inf_ssm", train=False)
    theirs = cl_datasets.build_transform(False, inf_ssm_args())
    for seed in range(3):
        img = cifar_image(seed)
        torch.testing.assert_close(ours(img), theirs(img), rtol=0, atol=0)


def test_inf_ssm_train_matches_the_baseline(cl_datasets):
    ours = sc._build_transform("inf_ssm", train=True)
    theirs = cl_datasets.build_transform(True, inf_ssm_args())
    assert [type(t).__name__ for t in ours.transforms] == [type(t).__name__ for t in theirs.transforms]
    for seed in range(5):
        img = cifar_image(seed)
        a = seeded(lambda: ours(img), seed)
        b = seeded(lambda: theirs(img), seed)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_inf_ssm_posterize_handles_zero_bits():
    out = sc._posterize(cifar_image(), 0)
    assert np.asarray(out).max() == 0


# ── loader on a fake CIFAR-100 ───────────────────────────────────────────────

class FakeCIFAR100:
    """Stands in for torchvision's CIFAR100: 2 images per class, 32x32 RGB."""

    def __init__(self, root, train, download, transform):
        self.transform = transform
        self.train = train
        self.targets = [c for c in range(100) for _ in range(2)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(cifar_image(i)), self.targets[i]


@pytest.fixture
def fake_cifar(monkeypatch, tmp_path):
    for name in ("meta", "train", "test"):
        (tmp_path / "cifar-100-python").mkdir(exist_ok=True)
        (tmp_path / "cifar-100-python" / name).write_bytes(b"")
    monkeypatch.setattr(sc.datasets, "CIFAR100", FakeCIFAR100)
    sc._load_cifar100.cache_clear()
    yield tmp_path
    sc._load_cifar100.cache_clear()


def loader(root, split, augment):
    return sc.get_task_loaders(0, split, root, batch_size=4, num_workers=0, seed=0, augment=augment,
                               n_tasks=10, class_order="natural", normalize="imagenet")


@pytest.mark.parametrize("split", ["train", "test"])
def test_inf_ssm_loader_yields_224_images(fake_cifar, split):
    images, labels = next(iter(loader(fake_cifar, split, "inf_ssm")))
    assert images.shape == (4, 3, 224, 224)
    assert labels.min() >= 0 and labels.max() < 10


@pytest.mark.parametrize("split", ["train", "test"])
def test_default_loader_yields_32_images(fake_cifar, split):
    images, _ = next(iter(loader(fake_cifar, split, True)))
    assert images.shape == (4, 3, 32, 32)


def test_inf_ssm_test_set_uses_the_eval_transform(fake_cifar):
    ds = loader(fake_cifar, "test", "inf_ssm").dataset
    names = [type(t).__name__ for t in ds.base.transform.transforms]
    assert names == ["Resize", "CenterCrop", "ToTensor", "Normalize"]
    # Without inf_ssm the test set is the plain one, whatever the train augment is.
    assert loader(fake_cifar, "test", "cifar").dataset.base is loader(fake_cifar, "test", "none").dataset.base
