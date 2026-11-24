"""
CPU tests for MambaBackbone on Vim-small (arch="vim_small") and the shared head.

OWNER: Hasinthaka Piyumal

Vim-small runs on CPU through the pure-PyTorch fallbacks of baselines/mamba-cl/utils/vim_mamba.py,
so a tiny Vim (2 blocks, 32-d, 32x32 images) is built for real here: checkpoint loading, the
x_proj split, the trainable set and the shared head are all exercised on the real model code.
The full-size model with the authors' task-0 checkpoint is checked on the GPU smoke run.
"""
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("timm")

from hippocortex.cl.null_space_projector import NullSpaceProjector  # noqa: E402
from hippocortex.cl.stats_buffer import StatsBuffer  # noqa: E402
from hippocortex.data.split_cifar100 import get_task_classes  # noqa: E402
from hippocortex.models import backbone as backbone_module  # noqa: E402
from hippocortex.models.backbone import MambaBackbone, _build_vim_small  # noqa: E402
from hippocortex.models.fake_defocus import TinyDefocus  # noqa: E402
from hippocortex.models.swr_generator import SWRGenerator  # noqa: E402
from hippocortex.training.trainer import Trainer  # noqa: E402

TINY = dict(embed_dim=32, depth=2)  # passed through _build_vim_small to VisionMamba
IMG = 32
N_TOTAL = 20
PER_TASK = 10


def vim_module():
    return backbone_module._load_mamba_cl_module("vim_mamba")


def save_tiny_checkpoint(path, num_classes: int = N_TOTAL, seed: int = 0) -> dict:
    """A Vim checkpoint as the authors store it: unsplit x_proj, under the 'model' key."""
    torch.manual_seed(seed)
    vim = vim_module()
    ref = vim.VisionMamba(img_size=IMG, patch_size=16, num_classes=num_classes,
                          drop_path_rate=0.0, **TINY)
    state = {k: v.clone() for k, v in ref.state_dict().items()}
    torch.save({"model": state}, path)
    return state


def tiny_backbone(ckpt=None, head_mode: str = "shared") -> MambaBackbone:
    model = _build_vim_small(ckpt, num_classes=N_TOTAL, img_size=IMG, **TINY)
    return MambaBackbone(model=model, image_size=IMG, head_mode=head_mode, n_classes_total=N_TOTAL)


def images(n: int = 3, seed: int = 1) -> torch.Tensor:
    return torch.randn(n, 3, IMG, IMG, generator=torch.Generator().manual_seed(seed))


# ── building and loading ─────────────────────────────────────────────────────

def test_checkpoint_loads_and_x_proj_is_split(tmp_path):
    state = save_tiny_checkpoint(tmp_path / "task0.pth")
    bb = tiny_backbone(tmp_path / "task0.pth")
    mixer = bb.model.layers[0].mixer
    assert mixer.is_split
    d_state = mixer.d_state
    torch.testing.assert_close(mixer.x_proj_C.weight, state["layers.0.mixer.x_proj.weight"][-d_state:])
    torch.testing.assert_close(bb.model.pos_embed, state["pos_embed"])


def test_shared_head_is_the_checkpoint_head(tmp_path):
    state = save_tiny_checkpoint(tmp_path / "task0.pth")
    bb = tiny_backbone(tmp_path / "task0.pth")
    torch.testing.assert_close(bb.shared_head.weight, state["head.weight"])
    assert isinstance(bb.model.head, torch.nn.Identity)


def test_checkpoint_head_of_another_size_is_replaced(tmp_path):
    save_tiny_checkpoint(tmp_path / "in1k.pth", num_classes=7)
    bb = tiny_backbone(tmp_path / "in1k.pth")  # load_finetune drops the 7-way head
    assert bb.shared_head.out_features == N_TOTAL


def test_missing_backbone_weights_raise(tmp_path):
    state = save_tiny_checkpoint(tmp_path / "ok.pth")
    torch.save({"model": {k: v for k, v in state.items() if not k.startswith("layers.")}},
               tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="did not load"):
        tiny_backbone(tmp_path / "bad.pth")


def test_missing_checkpoint_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tiny_backbone(tmp_path / "nope.pth")


def test_hf_checkpoint_path_downloads_the_task0_checkpoint(monkeypatch, tmp_path):
    save_tiny_checkpoint(tmp_path / "task0.pth")
    calls = []

    def fake_download(hf_dir, seed):
        calls.append((hf_dir, seed))
        return str(tmp_path / "task0.pth")

    def small_build(path, num_classes, img_size, grad_ckpt):
        return _build_vim_small(path, num_classes=num_classes, img_size=img_size, **TINY)

    monkeypatch.setattr(backbone_module, "task0_checkpoint", fake_download)
    monkeypatch.setattr(backbone_module, "_build_vim_small", small_build)
    bb = MambaBackbone(checkpoint_path="hf", image_size=IMG, head_mode="shared",
                       n_classes_total=N_TOTAL, hf_dir="cifar100_10task", seed=0)
    assert calls == [("cifar100_10task", 0)]
    assert bb.feature_dim == TINY["embed_dim"]


def test_trainable_set_is_mamba_cl_training_string():
    bb = tiny_backbone()
    trainable = {n for n, p in bb.model.named_parameters() if p.requires_grad}
    assert trainable, "no backbone parameter is trainable"
    patterns = ("x_proj", "out_proj.weight", "A_log", "A_b_log")
    assert all(any(pat in n for pat in patterns) for n in trainable)
    assert any(n.endswith("A_b_log") for n in trainable)  # Vim's backward A
    assert any("x_proj_step" in n for n in trainable)     # split before freezing
    assert not any("in_proj" in n or "norm" in n or "pos_embed" in n for n in trainable)


def test_hidden_is_the_class_token_feature():
    bb = tiny_backbone().eval()
    x = images()
    with torch.no_grad():
        torch.testing.assert_close(bb.extract_hidden(x), bb.model.forward_features(x))


# ── shared head ──────────────────────────────────────────────────────────────

def test_shared_head_grows_with_the_seen_classes():
    bb = tiny_backbone().eval()
    head0 = bb.set_task_head(PER_TASK)
    logits, hidden = bb(images())
    assert logits.shape == (3, 10) and hidden.shape == (3, 32)
    assert bb.set_task_head(PER_TASK) is head0  # one head for every task
    logits, _ = bb(images(), task_id=1)
    assert logits.shape == (3, 20)
    assert bb.label_offset(1) == 10 and bb.class_offset(1) == 10
    with torch.no_grad():
        full = bb.shared_head(bb.extract_hidden(images()))
        torch.testing.assert_close(bb.joined_logits(bb.extract_hidden(images())), full[:, :20])
        torch.testing.assert_close(bb.task_logits(bb.extract_hidden(images()), 1), full[:, 10:20])


def test_shared_head_rejects_more_classes_than_it_has():
    bb = tiny_backbone()
    bb.set_task_head(PER_TASK)
    bb.set_task_head(PER_TASK)
    with pytest.raises(ValueError, match="shared head"):
        bb.set_task_head(PER_TASK)


def test_shared_mode_needs_the_class_count():
    with pytest.raises(ValueError, match="n_classes_total"):
        MambaBackbone(model=TinyDefocus(embed_dim=32), head_mode="shared")


def test_per_task_mode_keeps_local_labels():
    bb = MambaBackbone(model=TinyDefocus(embed_dim=32), image_size=64)
    bb.set_task_head(5)
    bb.set_task_head(5)
    assert bb.label_offset(1) == 0 and bb.class_offset(1) == 5


# ── trainer with a shared head ───────────────────────────────────────────────

def shared_trainer(**training) -> Trainer:
    torch.manual_seed(0)
    bb = MambaBackbone(model=TinyDefocus(embed_dim=32, depth=2), image_size=64,
                       head_mode="shared", n_classes_total=N_TOTAL)
    t = dict(lr=1e-2, head_lr_scale=10.0, weight_decay=0.1, head_weight_decay=0.0,
             betas=[0.9, 0.999], epochs_warmup=1, epochs_joint=1, grad_clip=1.0,
             kl_weight=1e-3, use_amp=False, consolidate=False)
    t.update(training)
    cfg = OmegaConf.create({"dataset": {"n_classes_per_task": PER_TASK}, "training": t})
    swr = SWRGenerator(hidden_dim=32, latent_dim=8, n_tasks=2)
    return Trainer(bb, swr, NullSpaceProjector(rank_budget=8), StatsBuffer(), cfg)


def loader(label: int, n: int = 8) -> DataLoader:
    x = torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(label))
    return DataLoader(TensorDataset(x, torch.full((n,), label)), batch_size=4)


def test_shared_head_trains_on_global_labels():
    """Task 1's local label 3 is global class 13: its logit must rise, not class 3's."""
    tr = shared_trainer(epochs_joint=0, epochs_warmup=3)
    tr.train_task(0, loader(3))
    with torch.no_grad():
        before = tr.backbone.shared_head.bias.clone()
    tr.train_task(1, loader(3))
    delta = tr.backbone.shared_head.bias.detach() - before
    assert delta[13] > 0
    assert delta[13] == delta[:20].max()


def test_head_weight_decay_comes_from_its_own_setting():
    tr = shared_trainer()
    tr._start_task(0)
    wd = [g["weight_decay"] for g in tr.optimizer.param_groups]
    assert wd == [0.1, 0.0]


def test_first_task_training_can_be_skipped():
    tr = shared_trainer(train_first_task=False)
    head = tr.backbone.shared_head.weight.detach().clone()
    params = [p.detach().clone() for p in tr.backbone.backbone_parameters()]
    metrics = tr.train_task(0, loader(1))
    assert metrics["loss_warmup"] == 0.0 and metrics["loss_joint"] == 0.0
    torch.testing.assert_close(tr.backbone.shared_head.weight, head)
    for p, p0 in zip(tr.backbone.backbone_parameters(), params):
        torch.testing.assert_close(p, p0)
    tr.train_task(1, loader(1))  # task 1 still trains
    assert not torch.equal(tr.backbone.shared_head.weight, head)


def test_class_il_eval_with_a_shared_head_uses_global_indices():
    tr = shared_trainer(epochs_joint=0, epochs_warmup=0)
    tr._start_task(0)
    tr._start_task(1)
    with torch.no_grad():
        tr.backbone.shared_head.weight.zero_()
        tr.backbone.shared_head.bias.zero_()
        tr.backbone.shared_head.bias[12] = 5.0  # task 1, local label 2
    assert tr.evaluate(1, loader(2)) == 1.0
    assert tr.evaluate(1, loader(2), mode="task_il") == 1.0
    assert tr.evaluate(0, loader(2)) == 0.0  # global 2 is not predicted


# ── data split ───────────────────────────────────────────────────────────────

def test_natural_ten_task_split():
    split = get_task_classes(n_tasks=10, class_order="natural")
    assert len(split) == 10
    assert split[0] == list(range(10)) and split[9] == list(range(90, 100))


def test_split_rejects_task_counts_that_do_not_divide_100():
    with pytest.raises(ValueError, match="divide"):
        get_task_classes(n_tasks=7)


def test_default_split_is_unchanged():
    split = get_task_classes(42)
    assert len(split) == 20 and all(len(c) == 5 for c in split.values())
