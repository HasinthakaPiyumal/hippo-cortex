"""
CPU tests for scripts/extract_vim_feats.py and scripts/check_swr_on_vim_feats.py.

OWNER: Hasinthaka Piyumal

Extraction runs on a fake CIFAR-100 and a tiny backbone (the real Vim-small needs ~1 s per image
on CPU); the SWR check runs on synthetic class-clustered features.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("sklearn")
pytest.importorskip("timm")

from hippocortex.models.backbone import MambaBackbone  # noqa: E402
from hippocortex.models.fake_defocus import TinyDefocus  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


extract = load_script("extract_vim_feats")
check = load_script("check_swr_on_vim_feats")


# ── extract_vim_feats ────────────────────────────────────────────────────────

def test_task_indices_cover_the_task_and_respect_the_limit():
    targets = [c for _ in range(5) for c in range(100)]  # 5 images per class
    idx = extract.task_indices(targets, 2, None)
    assert {targets[i] for i in idx} == set(range(20, 30)) and len(idx) == 50
    idx = extract.task_indices(targets, 2, 20)
    assert len(idx) == 20 and {targets[i] for i in idx} == set(range(20, 30))


class FakeCIFAR100:
    def __init__(self, root, train, download, transform):
        self.transform = transform
        per_class = 3 if train else 2
        self.targets = [c for c in range(100) for _ in range(per_class)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        img = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        return self.transform(img), self.targets[i]


def test_extract_writes_features_meta_and_readme(monkeypatch, tmp_path):
    import torchvision

    def tiny_backbone(checkpoint_path, head_mode, n_classes_total):
        torch.manual_seed(0)
        return MambaBackbone(model=TinyDefocus(embed_dim=16, depth=1), image_size=64,
                             head_mode=head_mode, n_classes_total=n_classes_total)

    monkeypatch.setattr(torchvision.datasets, "CIFAR100", FakeCIFAR100)
    monkeypatch.setattr(extract, "MambaBackbone", tiny_backbone)
    monkeypatch.setattr(extract, "download_if_missing", lambda root: None)
    out = tmp_path / "feats"
    stats = extract.main(["--tasks", "0", "3", "--device", "cpu", "--num-workers", "0",
                          "--batch-size", "8", "--checkpoint", "local.pth", "--out", str(out)])
    train = torch.load(out / "train.pt", weights_only=True)
    test = torch.load(out / "test.pt", weights_only=True)
    assert train["feats"].dtype == torch.float16 and train["feats"].shape == (60, 16)
    assert test["feats"].shape == (40, 16)
    assert set(train["task_ids"].tolist()) == {0, 3}
    assert set(train["labels"][train["task_ids"] == 3].tolist()) == set(range(30, 40))
    meta = json.loads((out / "meta.json").read_text())
    assert meta["feature_dim"] == 16 and meta["tasks"] == [0, 3]
    assert set(stats["linear_probe_acc"]) == {0, 3}
    readme = (out / "README.md").read_text()
    assert "FROZEN" in readme.upper() and "| 3 | 30 | 20 |" in readme


# ── check_swr_on_vim_feats ───────────────────────────────────────────────────

def synthetic_feats(path: Path, d: int = 16, n_tasks: int = 2, per_class=(40, 10)):
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(10 * n_tasks, d, generator=g) * 3
    for split, k in zip(("train", "test"), per_class):
        feats, labels, tids = [], [], []
        for c in range(10 * n_tasks):
            feats.append(centres[c] + torch.randn(k, d, generator=g))
            labels += [c] * k
            tids += [c // 10] * k
        torch.save({"feats": torch.cat(feats).half(), "labels": torch.tensor(labels),
                    "task_ids": torch.tensor(tids)}, path / f"{split}.pt")


def test_swr_check_runs_and_reports(tmp_path):
    synthetic_feats(tmp_path)
    result = check.main(["--feats", str(tmp_path), "--tasks", "0", "1", "--epochs", "15",
                         "--batch-size", "64", "--lr", "1e-3"])
    assert result["stats_check"]["ok"]
    assert result["stats_check"]["memory_bytes"] == 2 * 16 * 2 * 4
    assert result["elbo_decreased"]
    rows = result["linear_probe"]
    assert set(rows) == {"real", "cvae", "rough", "per_class"}
    assert rows["real"]["acc"] > 0.9  # well-separated clusters
    assert rows["per_class"]["memory_bytes"] == 10 * 16 * 2 * 4
    assert json.loads((tmp_path / "swr_check.json").read_text())["train_task"] == 0


def test_swr_check_rejects_a_missing_task(tmp_path):
    synthetic_feats(tmp_path, n_tasks=1)
    with pytest.raises(SystemExit, match="not in"):
        check.main(["--feats", str(tmp_path), "--train-task", "4", "--epochs", "1"])
