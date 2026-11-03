"""CPU tests for Mamba-CL under the Inf-SSM protocol (baselines/inf-ssm/mamba_cl.py, --cl_method mamba_cl).

The projected update must not move a parameter along the input directions of the previous tasks, every
other backbone parameter must stay frozen, and the covariances must survive a checkpoint round trip.
"""
import argparse
import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("timm")
pytest.importorskip("scipy")

INF_SSM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../baselines/inf-ssm"))
if INF_SSM_DIR not in sys.path:
    sys.path.insert(0, INF_SSM_DIR)
import mamba_cl  # noqa: E402
import train_eval  # noqa: E402

DEPTH = 2
TINY = dict(img_size=32, depth=DEPTH, num_classes=4, drop_path_rate=0.)


class _Images(TensorDataset):
    transform = None  # eval_view() sets it; the tensors are already "transformed"


def _model(seed=0):
    torch.manual_seed(seed)
    vim, _ = mamba_cl.load_vim_module()
    model = vim.VisionMamba(patch_size=16, embed_dim=32, **TINY)  # Vim layout, tiny widths
    model.reload_params()
    for n, p in model.named_parameters():
        p.requires_grad_(any(s in n for s in mamba_cl.TRAIN_PATTERNS))
    return model


def _data(n=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return _Images(torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 2, (n,), generator=g))


def _collected(eta=1.0):
    model = _model()
    ns = mamba_cl.NullSpace(model, eta)
    ns.collect(model, _data(), batch_size=8, workers=0, device=torch.device("cpu"))
    return model, ns


def test_build_model_trains_only_mamba_cl_parameters():
    vim, _ = mamba_cl.load_vim_module()
    torch.manual_seed(0)
    model = mamba_cl.build_model("vim_small", None, **TINY)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert "head.weight" in trainable and "head.bias" in trainable
    for i in range(DEPTH):
        m = f"layers.{i}.mixer."
        for sfx in ("", "_b"):
            assert {f"{m}x_proj{sfx}_step.weight", f"{m}x_proj{sfx}_B.weight", f"{m}x_proj{sfx}_C.weight"} <= trainable
        assert {f"{m}A_log", f"{m}A_b_log", f"{m}out_proj.weight"} <= trainable
    frozen = {n for n, _ in model.named_parameters()} - trainable
    assert any("in_proj" in n for n in frozen) and any("conv1d" in n for n in frozen) and any("dt_proj" in n for n in frozen)
    assert all(n.startswith("head.") or "x_proj" in n or "A_log" in n or "A_b_log" in n or "out_proj.weight" in n for n in trainable)


def test_collect_covariance_per_projected_parameter():
    model, ns = _collected()
    named = dict(model.named_parameters())
    assert len(ns.cov) == 9 * DEPTH  # step, B, C, A per direction + out_proj
    for pname, c in ns.cov.items():
        w = named[pname]
        dim = w.shape[0] if ns.side[pname] == "left" else w.shape[1]
        assert c.shape == (dim, dim)
        assert torch.allclose(c, c.T, atol=1e-5)
        assert torch.linalg.eigvalsh(c.double()).min() > -1e-4
        assert 0 < ns.null_dims[pname] <= dim
    # collection must not leave the model in a different state
    assert all(p.requires_grad == any(s in n for s in mamba_cl.TRAIN_PATTERNS) for n, p in model.named_parameters())


def test_projected_update_is_orthogonal_to_old_inputs():
    model, ns = _collected(eta=1.0)
    plist = ns.params(model)
    before = ns.snapshot(plist)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p, _, _ in plist:
            p.add_(torch.randn(p.shape, generator=g))
    ns.project_(plist, before)
    named = {id(p): n for n, p in model.named_parameters()}
    for (p, _, side), p0 in zip(plist, before):
        pname = named[id(p)]
        _, S, Vt = torch.linalg.svd(ns.cov[pname])
        kept = Vt[~mamba_cl.adaptive_threshold(S)]  # the directions the old tasks use
        delta = p.detach() - p0
        moved = kept @ delta if side == "left" else delta @ kept.T
        assert moved.abs().max() < 1e-4 * max(delta.abs().max().item(), 1.0), pname


def test_null_space_checkpoint_round_trip():
    model, ns = _collected(eta=0.95)
    other = mamba_cl.NullSpace(model, 0.95)
    other.load_state_dict(ns.state_dict(), torch.device("cpu"))
    assert set(other.proj) == set(ns.proj)
    for n in ns.proj:
        assert torch.allclose(other.proj[n], ns.proj[n])
    with pytest.raises(AssertionError):
        mamba_cl.NullSpace(model, 0.9).load_state_dict(ns.state_dict(), torch.device("cpu"))


def test_train_one_epoch_projects_and_keeps_frozen_weights():
    model, ns = _collected(eta=1.0)
    R = train_eval.Run()
    R.args = argparse.Namespace(reg_lambda=0., reg_mode="inf_ssm", n_classes_per_task=2, nb_classes=4, progress_interval=0,
                                micro_batch=0, use_amp=False, clip_grad=None, smoothing=0.1, data_parallel=False)
    R.null_space = ns
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    data = _data()
    loader = DataLoader(TensorDataset(data.tensors[0], data.tensors[1] + 2), batch_size=8)  # task 1 = classes 2, 3
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    train_eval.train_one_epoch(R, 1, 0, model, None, loader, optimizer, train_eval.SoftTargetCrossEntropy(), None, scaler)

    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, start[n]), f"frozen {n} changed"
    moved_any = False
    for pname, side in ns.side.items():
        delta = dict(model.named_parameters())[pname].detach() - start[pname]
        moved_any |= delta.abs().max().item() > 0
        _, S, Vt = torch.linalg.svd(ns.cov[pname])
        kept = Vt[~mamba_cl.adaptive_threshold(S)]
        moved = kept @ delta if side == "left" else delta @ kept.T
        assert moved.abs().max() < 1e-4 * max(delta.abs().max().item(), 1e-6) + 1e-7, pname
    assert moved_any
    assert not torch.equal(model.head.weight, start["head.weight"])  # the head is not projected


def test_mamba_cl_optim_settings_and_head_lr(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_eval.py", "-d", "cifar100", "-t", "10", "--cl_method", "mamba_cl", "--optim", "mamba_cl"])
    args = train_eval.get_args()
    assert (args.lr, args.opt, args.weight_decay, args.task_lr_scaling, args.warmup_epochs) == (2e-4, "adam", 5e-5, 1.0, 0)
    assert train_eval.task_lr(args, 7) == 2e-4  # no per-task scaling
    assert "_mclopt_lr0.0002_" in train_eval.run_name(args)

    model = _model()
    optimizer, scheduler = train_eval.build_optimizer(args, model, train_eval.task_lr(args, 1))
    backbone, head = optimizer.param_groups
    assert backbone["lr"] == pytest.approx(2e-4) and head["lr"] == pytest.approx(2e-4 * 50)
    named = {id(p): n for n, p in model.named_parameters()}
    assert {named[id(p)] for p in head["params"]} == {"head.weight", "head.bias"}
    assert len(backbone["params"]) + 2 == sum(p.requires_grad for p in model.parameters())
    lrs = []
    for epoch in range(args.epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step(epoch)  # as train_one_task (and Mamba-CL): after each epoch, with its index
    assert lrs[0] == pytest.approx(2e-4) and lrs[19] == pytest.approx(2e-4)  # no warmup, flat to epoch 20
    assert lrs[20] == pytest.approx(2e-5) and lrs[31] == pytest.approx(2e-5) and lrs[32] == pytest.approx(2e-6)


def test_default_optim_is_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_eval.py", "-d", "cifar100", "-t", "10", "--cl_method", "mamba_cl"])
    args = train_eval.get_args()
    assert (args.optim, args.lr, args.opt, args.task_lr_scaling) == ("inf_ssm", 1e-5, "adamw", 0.5)
    assert "_mclopt" not in train_eval.run_name(args)
