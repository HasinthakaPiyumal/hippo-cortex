"""
Unit tests for MambaBackbone (De-focus Mamba base wrapper).

OWNER: Praveen Dedigama

CPU only. The real network needs the mamba-ssm / causal-conv1d CUDA kernels to be
constructed, so a small stand-in with the same parameter names (TinyDefocus) is injected
through ``MambaBackbone(model=...)``. The real model is covered by
tests/models/test_backbone_gpu.py and scripts/smoke_defocus_backbone.py.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hippocortex.models import backbone as backbone_module
from hippocortex.models.backbone import (
    DEFAULT_CONFIG_PATH,
    MambaBackbone,
    check_load_report,
)
from hippocortex.models.fake_defocus import TinyDefocus

D = 32          # TinyDefocus embed_dim
IMG = 64        # image_size used in tests (TinyDefocus patch=32 -> 4 patches)


def make_backbone(depth: int = 2) -> MambaBackbone:
    return MambaBackbone(model=TinyDefocus(embed_dim=D, depth=depth), image_size=IMG)


def images(batch: int = 3, size: int = 32) -> torch.Tensor:
    return torch.randn(batch, 3, size, size)


# ── forward / extract_hidden ────────────────────────────────────────────────

def test_forward_output_shapes():
    """forward() returns logits (B, n_classes) and hidden (B, feature_dim)."""
    bb = make_backbone()
    bb.set_task_head(5)
    logits, hidden = bb(images(3))
    assert logits.shape == (3, 5)
    assert hidden.shape == (3, D)
    assert bb.feature_dim == D


def test_forward_resizes_images_to_image_size():
    """Any HxW input is resized before it reaches encode_image."""
    seen = []

    class Spy(TinyDefocus):
        def encode_image(self, x, *args, **kwargs):
            seen.append(tuple(x.shape))
            return super().encode_image(x, *args, **kwargs)

    bb = MambaBackbone(model=Spy(embed_dim=D), image_size=IMG)
    bb.set_task_head(5)
    bb(images(2, 32))
    bb(images(2, IMG))
    assert seen == [(2, 3, IMG, IMG), (2, 3, IMG, IMG)]


def test_forward_rejects_non_image_input():
    bb = make_backbone()
    bb.set_task_head(5)
    with pytest.raises(ValueError, match="B, 3, H, W"):
        bb(torch.randn(2, 16, D))


def test_extract_hidden_matches_forward():
    """extract_hidden() is the hidden returned by forward() (same extraction point)."""
    bb = make_backbone().eval()
    bb.set_task_head(5)
    x = images(2)
    with torch.no_grad():
        _, hidden_fwd = bb(x)
        hidden_ext = bb.extract_hidden(x)
    assert hidden_ext.shape == (2, D)
    assert torch.allclose(hidden_fwd, hidden_ext, atol=1e-6)


def test_extract_hidden_needs_no_head():
    assert make_backbone().extract_hidden(images(2)).shape == (2, D)


def test_encode_image_with_wrong_feature_shape_is_rejected():
    class Bad(TinyDefocus):
        def encode_image(self, x, *args, **kwargs):
            return torch.zeros(x.shape[0], 3, D)  # tokens, not pooled features

    bb = MambaBackbone(model=Bad(embed_dim=D), image_size=IMG)
    with pytest.raises(ValueError, match="expected"):
        bb.extract_hidden(images(2))


# ── per-task heads ──────────────────────────────────────────────────────────

def test_set_task_head_keeps_old_heads_and_their_outputs():
    """Adding a head must not replace or change earlier heads."""
    bb = make_backbone().eval()
    bb.set_task_head(5)
    x = images(2)
    with torch.no_grad():
        logits_before, _ = bb(x, task_id=0)
    w0 = bb.heads["0"].weight.clone()

    new_head = bb.set_task_head(3)

    assert bb.n_tasks == 2
    assert bb.heads["1"] is new_head
    assert torch.equal(bb.heads["0"].weight, w0)
    with torch.no_grad():
        logits_old, _ = bb(x, task_id=0)
        logits_new, _ = bb(x)           # default = latest head
    assert torch.equal(logits_old, logits_before)
    assert logits_old.shape == (2, 5)
    assert logits_new.shape == (2, 3)


def test_set_task_head_returns_linear_on_model_device_and_dtype():
    bb = make_backbone()
    ref = next(bb.model.parameters())
    head = bb.set_task_head(7)
    assert isinstance(head, torch.nn.Linear)
    assert head.in_features == D and head.out_features == 7
    assert head.weight.device == ref.device and head.weight.dtype == ref.dtype


def test_set_task_head_with_explicit_id_recreates_that_head():
    bb = make_backbone()
    bb.set_task_head(5, task_id=0)
    first = bb.heads["0"]
    bb.set_task_head(5, task_id=0)
    assert bb.n_tasks == 1
    assert bb.heads["0"] is not first


def test_set_task_head_rejects_gaps():
    bb = make_backbone()
    with pytest.raises(ValueError, match="gap"):
        bb.set_task_head(5, task_id=2)


def test_forward_without_head_and_with_unknown_task():
    bb = make_backbone()
    with pytest.raises(RuntimeError, match="set_task_head"):
        bb(images(1))
    bb.set_task_head(5)
    with pytest.raises(KeyError):
        bb(images(1), task_id=3)


def test_only_current_head_is_trainable():
    bb = make_backbone()
    bb.set_task_head(5)
    bb.set_task_head(5)
    assert not any(p.requires_grad for p in bb.heads["0"].parameters())
    assert all(p.requires_grad for p in bb.heads["1"].parameters())
    assert bb.head_parameters(1) == list(bb.heads["1"].parameters())


def test_imagenet_head_is_removed():
    bb = make_backbone()
    assert not any(n.startswith("model.head.") for n, _ in bb.named_parameters())


# ── training scope (Mamba-CL --training_string) ─────────────────────────────

def test_only_mamba_cl_parameters_are_trainable():
    bb = make_backbone(depth=2)
    bb.set_task_head(5)
    trainable = {n for n, p in bb.named_parameters() if p.requires_grad}
    expected = {"heads.0.weight", "heads.0.bias"}
    for i in range(2):
        m = f"model.blocks.{i}.mixer."
        expected |= {
            m + "A_log", m + "x_proj_step.weight", m + "x_proj_B.weight",
            m + "x_proj_C.weight", m + "out_proj.weight",
        }
    assert trainable == expected
    frozen = {n for n, p in bb.named_parameters() if not p.requires_grad}
    assert "model.patch_embed.proj.weight" in frozen
    assert "model.cls_token" in frozen
    assert "model.norm.weight" in frozen
    assert "model.blocks.0.mixer.in_proj.weight" in frozen


def test_backbone_parameters_excludes_heads_and_frozen():
    bb = make_backbone(depth=2)
    bb.set_task_head(5)
    assert len(bb.backbone_parameters()) == 2 * 5
    head_ids = {id(p) for p in bb.head_parameters()}
    assert not head_ids & {id(p) for p in bb.backbone_parameters()}


def test_gradient_flow_reaches_exactly_the_trainable_parameters():
    bb = make_backbone()
    bb.set_task_head(4)
    logits, hidden = bb(images(2))
    (logits.sum() + hidden.sum()).backward()
    for name, p in bb.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} got no gradient"
            assert not torch.isnan(p.grad).any(), f"{name} has NaN gradient"
            assert p.grad.abs().sum() > 0, f"{name} gradient is identically zero"
        else:
            assert p.grad is None, f"frozen {name} received a gradient"


def test_hidden_depends_on_the_image():
    """Guards the stand-in itself: a constant hidden would make every training test vacuous."""
    bb = make_backbone().eval()
    with torch.no_grad():
        h1, h2 = bb.extract_hidden(images(2)), bb.extract_hidden(images(2))
    assert not torch.allclose(h1, h2, atol=1e-4)


# ── loading the real checkpoint (logic only; no CUDA kernels needed) ─────────

def test_check_load_report_flags_missing_backbone_weights(capsys):
    check_load_report(["head.weight", "head.bias"], ["something.else"])  # fine
    assert "missing keys" in capsys.readouterr().out

    for key in ("blocks.3.mixer.out_proj.weight", "patch_embed.proj.weight", "norm.weight", "cls_token"):
        with pytest.raises(RuntimeError, match="did not load"):
            check_load_report([key], [])


class _RecordingTiny(TinyDefocus):
    calls: list = []

    def load_state_dict(self, *args, **kwargs):
        type(self).calls.append("load_state_dict")
        return super().load_state_dict(*args, **kwargs)

    def reload_params(self):
        type(self).calls.append("reload_params")
        super().reload_params()


@pytest.fixture
def fake_build(monkeypatch):
    """Route _build_defocus_model to a stand-in instead of the CUDA-only real model."""
    _RecordingTiny.calls = []
    cfg_module = SimpleNamespace(
        get_config=lambda path: path,
        build_model=lambda cfg: _RecordingTiny(embed_dim=D),
    )
    monkeypatch.setattr(backbone_module, "_load_mamba_cl_config_module", lambda: cfg_module)
    return _RecordingTiny


def test_build_loads_checkpoint_before_reload_params(tmp_path, fake_build):
    """reload_params() splits x_proj, which the checkpoint stores unsplit, so it must come last."""
    ckpt = tmp_path / "ckpt.pth"
    torch.save({"model": TinyDefocus(embed_dim=D).state_dict()}, ckpt)

    bb = MambaBackbone(checkpoint_path=ckpt, image_size=IMG)

    assert fake_build.calls == ["load_state_dict", "reload_params"]
    assert bb.feature_dim == D


def test_build_fails_when_backbone_weights_are_missing(tmp_path, fake_build):
    state = {k: v for k, v in TinyDefocus(embed_dim=D).state_dict().items() if not k.startswith("blocks.")}
    ckpt = tmp_path / "ckpt.pth"
    torch.save({"model": state}, ckpt)
    with pytest.raises(RuntimeError, match="did not load"):
        MambaBackbone(checkpoint_path=ckpt)


def test_build_accepts_a_bare_state_dict(tmp_path, fake_build):
    ckpt = tmp_path / "ckpt.pth"
    torch.save(TinyDefocus(embed_dim=D).state_dict(), ckpt)
    MambaBackbone(checkpoint_path=ckpt)
    assert fake_build.calls == ["load_state_dict", "reload_params"]


def test_missing_checkpoint_error_names_the_download_url(tmp_path, fake_build):
    with pytest.raises(FileNotFoundError, match="defocus_mamba_base_cls.pth"):
        MambaBackbone(checkpoint_path=tmp_path / "nope.pth")


def test_constructor_needs_a_checkpoint_or_a_model():
    with pytest.raises(ValueError, match="checkpoint_path"):
        MambaBackbone()


# ── baseline code is importable and configured as expected ───────────────────

_BASELINE_DEPS = ("timm", "yacs", "easydict", "transformers")
_have_baseline_deps = all(importlib.util.find_spec(m) for m in _BASELINE_DEPS)


@pytest.mark.skipif(not _have_baseline_deps, reason="needs timm, yacs, easydict, transformers")
def test_baseline_config_describes_defocus_mamba_base():
    cfg_module = backbone_module._load_mamba_cl_config_module()
    cfg = cfg_module.get_config(str(DEFAULT_CONFIG_PATH))
    assert cfg.MODEL.NAME == "defocus_mamba_base"
    assert cfg.MODEL.depth == 24
    assert cfg.MODEL.MAMBA.intermediate_size == 1536
    assert cfg.DATA.IMG_SIZE == 224


@pytest.mark.skipif(
    not _have_baseline_deps or importlib.util.find_spec("mamba_ssm") is not None,
    reason="only meaningful without the CUDA kernels",
)
def test_real_model_without_cuda_kernels_gives_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="mamba-ssm"):
        MambaBackbone(checkpoint_path=tmp_path / "unused.pth")


def test_freezing_a_head_clears_its_stale_gradients():
    """A frozen head is in no optimizer, so a gradient left on it would never be cleared."""
    bb = make_backbone()
    bb.set_task_head(5)
    logits, _ = bb(images(2))
    logits.sum().backward()
    assert bb.heads["0"].weight.grad is not None

    bb.set_task_head(5)
    assert all(p.grad is None for p in bb.heads["0"].parameters())
