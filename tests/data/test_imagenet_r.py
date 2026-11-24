"""
CPU tests for the ImageNet-R loader, on a fake 200-class image folder (no download).

OWNER: Hasinthaka Piyumal

The Inf-SSM protocol is compared with baselines/inf-ssm/cl_datasets.py::build_cl_splits on the
same folder, and the Mamba-CL class split with its ClassIncrementalManager.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("timm")

from hippocortex.data import imagenet_r as ir  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
PER_CLASS = 4  # 800 images


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("inr")
    rng = np.random.default_rng(0)
    wnids = sorted(f"n{10000000 + 37 * c:08d}" for c in range(ir.N_CLASSES))
    for c, wnid in enumerate(wnids):
        d = root / "imagenet-r" / wnid
        d.mkdir(parents=True)
        for j in range(PER_CLASS):
            arr = rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"img_{j}.jpg")
    return root


def import_baseline(subdir: str, module: str):
    path = str(_REPO_ROOT / "baselines" / subdir)
    sys.path.insert(0, path)
    try:
        return __import__(module)
    finally:
        sys.path.remove(path)


# ── class splits ─────────────────────────────────────────────────────────────

def test_inf_ssm_split_is_natural_order():
    split = ir.get_task_classes("inf_ssm")
    assert len(split) == 10
    assert split[0] == list(range(20)) and split[9] == list(range(180, 200))


@pytest.mark.parametrize("protocol", ir.PROTOCOLS)
def test_tasks_are_disjoint_and_cover_200_classes(protocol):
    split = ir.get_task_classes(protocol)
    seen = [c for cs in split.values() for c in cs]
    assert sorted(seen) == list(range(200))
    assert len({len(cs) for cs in split.values()}) == 1


def test_mamba_cl_split_matches_its_class_manager():
    cm = import_baseline("mamba-cl/utils", "continual_manager")
    ref = cm.ClassIncrementalManager(list(range(200)), 20, 2024, shuffle=True).task_class_list
    ours = ir.get_task_classes("mamba_cl")
    assert [ours[t] for t in range(20)] == ref
    # The PCG64 recipe of the old docstring gives the same split.
    rng = np.random.Generator(np.random.PCG64(2024))
    classes = list(range(200))
    rng.shuffle(classes)
    assert np.array(classes).reshape(20, 10).tolist() == ref


def test_split_rejects_task_counts_that_do_not_divide_200():
    with pytest.raises(ValueError, match="divide"):
        ir.get_task_classes("inf_ssm", n_tasks=7)
    with pytest.raises(ValueError, match="protocol"):
        ir.get_task_classes("l2p")


def test_write_task_split(tmp_path):
    path = ir.write_task_split(tmp_path)
    assert path.name == "task_split_mamba_cl_seed2024.json"
    data = json.loads(path.read_text())
    assert data["0"] == ir.get_task_classes("mamba_cl")[0]


# ── Inf-SSM partition and loaders ────────────────────────────────────────────

def inf_ssm_args(root):
    return SimpleNamespace(
        dataset="imagenet_r", num_tasks=10, random_class=False, seed=0, data_root=str(root),
        input_size=224, eval_crop_ratio=0.875, color_jitter=0.3, aa="rand-m9-mstd0.5-inc1",
        train_interpolation="bicubic", reprob=0.25, remode="pixel", recount=1,
    )


def test_inf_ssm_partition_matches_the_baseline(fake_root):
    cl = import_baseline("inf-ssm", "cl_datasets")
    g = torch.Generator()
    g.manual_seed(0)
    train, val, test = cl.build_cl_splits(inf_ssm_args(fake_root / "imagenet-r"), g)
    for task in (0, 4, 9):
        for split, ref in (("train", train), ("val", val), ("test", test)):
            ref.set_task(task)
            ours = ir.get_task_dataset(task, split, fake_root)
            assert ours.indices == ref.cur_task_idx.tolist(), (task, split)
            # Labels: the baseline keeps global ids; ours are local to the task.
            _, y_ref = ref[0]
            _, y = ours[0]
            assert y == y_ref - 20 * task


def test_inf_ssm_partition_sizes(fake_root):
    parts = ir.inf_ssm_partition(ir.N_CLASSES * PER_CLASS)
    n = ir.N_CLASSES * PER_CLASS
    assert len(parts["test"]) == n - int(0.8 * n)
    assert len(parts["train"]) == int(0.9 * int(0.8 * n))
    allidx = parts["train"] + parts["val"] + parts["test"]
    assert sorted(allidx) == list(range(n))


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_inf_ssm_loader_output(fake_root, split):
    images, labels = next(iter(ir.get_task_loaders(3, split, fake_root, batch_size=4, num_workers=0)))
    assert images.shape == (4, 3, 224, 224)
    assert labels.min() >= 0 and labels.max() < 20


def test_inf_ssm_eval_transform_matches_the_baseline(fake_root):
    cl = import_baseline("inf-ssm", "cl_datasets")
    theirs = cl.build_transform(False, inf_ssm_args(fake_root))
    ours = ir.build_transform("inf_ssm", train=False)
    img = Image.open(next((fake_root / "imagenet-r").glob("*/*.jpg"))).convert("RGB")
    torch.testing.assert_close(ours(img), theirs(img), rtol=0, atol=0)


def test_root_can_be_the_image_folder_itself(fake_root):
    a = ir.get_task_dataset(0, "test", fake_root)
    b = ir.get_task_dataset(0, "test", fake_root / "imagenet-r")
    assert a.indices == b.indices
    assert ir.is_available(fake_root)


def test_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="ImageNet-R"):
        ir.get_task_dataset(0, "train", tmp_path / "nope")


# ── Mamba-CL partition ───────────────────────────────────────────────────────

def test_mamba_cl_split_files_cover_imagenet_r():
    names = {s: ir.MAMBA_CL_SPLIT_FILES[s].read_text().split() for s in ("train", "val")}
    assert len(names["train"]) == 24000 and len(names["val"]) == 6000
    assert not set(names["train"]) & set(names["val"])
    assert len({n.split("/")[0] for n in names["train"] + names["val"]}) == 200


def test_mamba_cl_loader_uses_its_split_files(fake_root, monkeypatch, tmp_path):
    files = sorted(p.relative_to(fake_root / "imagenet-r").as_posix()
                   for p in (fake_root / "imagenet-r").glob("*/*.jpg"))
    train_names, val_names = files[::2], files[1::2]
    for split, names in (("train", train_names), ("val", val_names)):
        (tmp_path / f"{split}.txt").write_text("\n".join(names) + "\n")
    monkeypatch.setattr(ir, "MAMBA_CL_SPLIT_FILES",
                        {"train": tmp_path / "train.txt", "val": tmp_path / "val.txt"})
    task_classes = ir.get_task_classes("mamba_cl")
    ds = ir.get_task_dataset(0, "train", fake_root, protocol="mamba_cl")
    folder = ds.folder
    got = {Path(folder.samples[i][0]).relative_to(folder.root).as_posix() for i in ds.indices}
    want = {n for n in train_names if folder.class_to_idx[n.split("/")[0]] in task_classes[0]}
    assert got == want and len(got) == 10 * PER_CLASS // 2
    test = ir.get_task_dataset(0, "test", fake_root, protocol="mamba_cl")
    assert test.indices == ir.get_task_dataset(0, "val", fake_root, protocol="mamba_cl").indices
    images, labels = next(iter(DataLoader_of(ds)))
    assert images.shape == (2, 3, 224, 224) and labels.max() < 10


def DataLoader_of(ds):
    return torch.utils.data.DataLoader(ds, batch_size=2)


def test_mamba_cl_missing_images_raise(fake_root, monkeypatch, tmp_path):
    (tmp_path / "train.txt").write_text("n00000000/missing.jpg\n")
    (tmp_path / "val.txt").write_text("")
    monkeypatch.setattr(ir, "MAMBA_CL_SPLIT_FILES",
                        {"train": tmp_path / "train.txt", "val": tmp_path / "val.txt"})
    with pytest.raises(FileNotFoundError, match="not under"):
        ir.get_task_dataset(0, "train", fake_root, protocol="mamba_cl")
