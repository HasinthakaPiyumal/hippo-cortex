"""
GPU tests for MambaBackbone with the REAL De-focus Mamba base weights.

OWNER: Praveen Dedigama

Skipped unless CUDA, mamba-ssm and the checkpoint are all available, so they never run
in CPU CI. On the Kaggle T4:

    DEFOCUS_CKPT=/kaggle/working/defocus_mamba_base_cls.pth pytest tests/models/test_backbone_gpu.py

scripts/smoke_defocus_backbone.py covers the same ground plus the Trainer, with a report.
"""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from hippocortex.models.backbone import MambaBackbone

CKPT = Path(os.environ.get("DEFOCUS_CKPT", "checkpoints/defocus_mamba_base_cls.pth"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("mamba_ssm") is None
    or not CKPT.is_file(),
    reason="needs CUDA, mamba-ssm and the defocus_mamba_base_cls.pth checkpoint",
)


@pytest.fixture(scope="module")
def backbone():
    bb = MambaBackbone(checkpoint_path=CKPT).cuda().eval()
    bb.set_task_head(5)
    return bb


def test_real_forward_shapes(backbone):
    x = torch.randn(2, 3, 32, 32, device="cuda")  # CIFAR-sized; resized to 224 inside
    with torch.no_grad():
        logits, hidden = backbone(x)
    assert logits.shape == (2, 5)
    assert hidden.shape == (2, 768)
    assert torch.isfinite(hidden).all()


def test_real_extract_hidden_matches_forward(backbone):
    x = torch.randn(2, 3, 224, 224, device="cuda")
    with torch.no_grad():
        _, hidden = backbone(x)
        assert torch.allclose(hidden, backbone.extract_hidden(x), atol=1e-5)


def test_real_trainable_scope(backbone):
    names = [n for n, p in backbone.model.named_parameters() if p.requires_grad]
    assert names and all(any(k in n for k in ("x_proj", "out_proj.weight", "A_log")) for n in names)
    assert not any("x_proj." in n for n in names), "reload_params() should have split x_proj"


def test_real_one_step_moves_only_trainable_parameters():
    bb = MambaBackbone(checkpoint_path=CKPT).cuda().train()
    bb.set_task_head(5)
    before = {n: p.detach().clone() for n, p in bb.named_parameters()}
    opt = torch.optim.AdamW([p for p in bb.parameters() if p.requires_grad], lr=1e-2)

    logits, _ = bb(torch.randn(4, 3, 32, 32, device="cuda"))
    torch.nn.functional.cross_entropy(logits, torch.randint(0, 5, (4,), device="cuda")).backward()
    opt.step()

    for n, p in bb.named_parameters():
        if p.requires_grad:
            continue
        assert torch.equal(p, before[n]), f"frozen {n} changed"
    assert any(not torch.equal(p, before[n]) for n, p in bb.named_parameters() if p.requires_grad)
