"""
Tests for the Trainer's per-task optimizer and Phase 2 (joint) loop.

OWNER: Praveen Dedigama

CPU only, with a TinyDefocus stand-in backbone, the real SWRGenerator / StatsBuffer /
NullSpaceProjector. consolidate() is not implemented yet, so tests play the role of the
consolidation phase by calling ``buffer.update`` and ``projector.update`` themselves.
"""
import logging
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from hippocortex.cl.null_space_projector import NullSpaceProjector
from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.models.backbone import MambaBackbone
from hippocortex.models.fake_defocus import TinyDefocus
from hippocortex.models.swr_generator import SWRGenerator
from hippocortex.training.trainer import Trainer

D = 32          # feature width (TinyDefocus embed_dim)
N_CLS = 5
DEPTH = 2
BATCH = 8


def make_cfg(**training) -> "OmegaConf":
    t = dict(
        lr=1e-2, head_lr_scale=10.0, weight_decay=0.0, betas=[0.9, 0.999],
        epochs_warmup=1, epochs_joint=1, grad_clip=1.0, kl_weight=1e-3, use_amp=False,
    )
    t.update(training)
    return OmegaConf.create({"dataset": {"n_classes_per_task": N_CLS}, "training": t})


def make_loader(n: int = 16, labels=None, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, N_CLS, (n,), generator=g) if labels is None else torch.full((n,), labels)
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=False)


def make_trainer(**training) -> Trainer:
    torch.manual_seed(0)
    backbone = MambaBackbone(model=TinyDefocus(embed_dim=D, depth=DEPTH), image_size=64)
    swr = SWRGenerator(hidden_dim=D, latent_dim=8, n_tasks=4)
    return Trainer(backbone, swr, NullSpaceProjector(rank_budget=16), StatsBuffer(), make_cfg(**training))


def consolidate_stub(trainer: Trainer, task_id: int, rank: int = 4) -> None:
    """Stand-in for Phase 3: store (mu, sigma2) and extend the null-space basis."""
    torch.manual_seed(100 + task_id)
    h = torch.randn(64, rank) @ torch.randn(rank, D)  # low-rank so U leaves a null space
    trainer.buffer.update(task_id, h)
    trainer.projector.update(h)


def run_tasks_until(trainer: Trainer, last_task: int) -> None:
    """train_task(0..last_task-1), calling the consolidation stand-in after each."""
    for t in range(last_task):
        trainer.train_task(t, make_loader(seed=t))
        consolidate_stub(trainer, t)


def backward_one_batch(trainer: Trainer, task_id: int) -> None:
    images, labels = next(iter(make_loader()))
    trainer.optimizer.zero_grad(set_to_none=True)
    logits, _ = trainer.backbone(images, task_id=task_id)
    trainer._criterion(logits, labels).backward()


def grad_norm(trainer: Trainer) -> float:
    grads = [p.grad for p in trainer.backbone.parameters() if p.grad is not None]
    return torch.sqrt(sum((g ** 2).sum() for g in grads)).item()


# ── task 0: nothing to replay, nothing to project ────────────────────────────

def test_task0_skips_replay_and_projection():
    tr = make_trainer()
    tr.swr_gen.sample = MagicMock(wraps=tr.swr_gen.sample)

    metrics = tr.train_task(0, make_loader())

    tr.swr_gen.sample.assert_not_called()
    assert metrics["replay_loss"] == 0.0
    assert metrics["proj_params_projected"] == 0 and metrics["proj_params_skipped"] == 0
    assert metrics["loss_joint"] > 0
    assert set(metrics) >= {"task_id", "loss_warmup", "loss_joint", "loss_consolidation", "replay_loss"}


# ── replay plumbing ──────────────────────────────────────────────────────────

def test_every_past_task_is_sampled_once_per_batch_with_batch_sized_n():
    tr = make_trainer()
    run_tasks_until(tr, 2)
    tr.swr_gen.sample = MagicMock(wraps=tr.swr_gen.sample)

    tr.train_task(2, make_loader())

    calls = tr.swr_gen.sample.call_args_list
    n_batches = 2  # 16 samples / BATCH, one joint epoch
    assert len(calls) == 2 * n_batches
    assert sorted({c.args[0] for c in calls}) == [0, 1]
    assert all(c.args[1] == BATCH for c in calls)
    mu, sigma2 = calls[0].args[2]
    assert mu.shape == (D,) and sigma2.shape == (D,)


def test_replay_only_uses_tasks_before_the_current_one():
    tr = make_trainer()
    run_tasks_until(tr, 1)
    consolidate_stub(tr, 1)            # buffer already holds task 1 ...
    tr.swr_gen.sample = MagicMock(wraps=tr.swr_gen.sample)
    tr.train_task(1, make_loader())    # ... but task 1 is being (re)trained
    assert {c.args[0] for c in tr.swr_gen.sample.call_args_list} == {0}


def test_replay_term_is_reported_and_carries_no_gradient():
    tr = make_trainer()
    run_tasks_until(tr, 1)
    gen_before = {k: v.clone() for k, v in tr.swr_gen.state_dict().items()}

    metrics = tr.train_task(1, make_loader())

    assert metrics["replay_loss"] > 0
    assert all(p.grad is None for p in tr.swr_gen.parameters()), "generator collected stray grads"
    for k, v in tr.swr_gen.state_dict().items():
        assert torch.equal(v, gen_before[k]), f"generator weight {k} changed in Phase 2"
    assert not tr._replay_loss(1, BATCH).requires_grad


def test_replay_loss_is_zero_without_past_tasks():
    tr = make_trainer()
    assert tr._replay_loss(0, BATCH).item() == 0.0


# ── projection ───────────────────────────────────────────────────────────────

def test_projection_covers_head_and_out_proj_and_skips_the_rest():
    tr = make_trainer()
    run_tasks_until(tr, 1)
    tr._start_task(1)
    backward_one_batch(tr, 1)

    head_w, head_b = tr.backbone.heads["1"].weight, tr.backbone.heads["1"].bias
    out_projs = [b.mixer.out_proj.weight for b in tr.backbone.model.blocks]
    skipped_before = {
        n: p.grad.clone() for n, p in tr.backbone.named_parameters()
        if p.grad is not None and ("x_proj" in n or "A_log" in n) or n == "heads.1.bias"
    }
    head_w_before = head_w.grad.clone()
    out_before = [w.grad.clone() for w in out_projs]

    n_projected, n_skipped = tr._project_gradients()

    U = tr.projector._U
    assert n_projected == 1 + DEPTH                 # head weight + one out_proj per block
    assert n_skipped == 1 + DEPTH * 4               # head bias + 3 x_proj + A_log per block

    # last-axis case: (n_classes, d_model) -> G (I - U U^T)
    assert (head_w.grad @ U).abs().max() < 1e-5
    assert torch.allclose(head_w.grad, head_w_before - (head_w_before @ U) @ U.T, atol=1e-6)
    assert not torch.allclose(head_w.grad, head_w_before)

    # first-axis case: (d_model, d_inner) -> (I - U U^T) G
    for w, before in zip(out_projs, out_before):
        assert (U.T @ w.grad).abs().max() < 1e-5
        assert torch.allclose(w.grad, before - U @ (U.T @ before), atol=1e-6)

    # shapes with no d_model axis are left alone
    for n, p in tr.backbone.named_parameters():
        if n in skipped_before:
            assert torch.equal(p.grad, skipped_before[n]), f"{n} was modified"


def test_no_projection_before_the_first_consolidation():
    tr = make_trainer()
    tr._start_task(0)
    backward_one_batch(tr, 0)
    before = {n: p.grad.clone() for n, p in tr.backbone.named_parameters() if p.grad is not None}

    assert tr._project_gradients() == (0, 0)

    for n, p in tr.backbone.named_parameters():
        if p.grad is not None:
            assert torch.equal(p.grad, before[n])


def test_skipped_parameters_warn_exactly_once(caplog):
    tr = make_trainer()
    run_tasks_until(tr, 1)
    tr._start_task(1)
    backward_one_batch(tr, 1)

    with caplog.at_level(logging.WARNING, logger="hippocortex.training.trainer"):
        tr._project_gradients()
        backward_one_batch(tr, 1)
        tr._project_gradients()

    warnings = [r for r in caplog.records if "projection skipped" in r.getMessage()]
    assert len(warnings) == 1
    assert "x_proj" in warnings[0].getMessage() or "A_log" in warnings[0].getMessage()


def test_joint_metrics_report_projected_and_skipped_counts():
    tr = make_trainer()
    run_tasks_until(tr, 1)
    metrics = tr.train_task(1, make_loader())
    assert metrics["proj_params_projected"] == 1 + DEPTH
    assert metrics["proj_params_skipped"] == 1 + DEPTH * 4


# ── clipping ─────────────────────────────────────────────────────────────────

def test_gradients_are_clipped_before_the_step():
    bound = 1e-3
    tr = make_trainer(grad_clip=bound)
    tr._start_task(0)

    backward_one_batch(tr, 0)
    assert grad_norm(tr) > bound * 10, "test needs unclipped grads well above the bound"

    seen = []
    original_step = tr.optimizer.step
    tr.optimizer.step = lambda *a, **k: (seen.append(grad_norm(tr)), original_step(*a, **k))[1]
    images, labels = next(iter(make_loader()))
    tr._joint_step(0, images, labels)

    assert len(seen) == 1
    assert seen[0] <= bound * (1 + 1e-4)


def test_clipping_can_be_disabled():
    tr = make_trainer(grad_clip=None)
    tr._start_task(0)
    images, labels = next(iter(make_loader()))
    total, replay, _, _ = tr._joint_step(0, images, labels)
    assert total > 0 and replay == 0.0


# ── optimizer / heads ────────────────────────────────────────────────────────

def test_new_head_is_in_the_optimizer_and_old_head_is_not():
    tr = make_trainer()
    tr.train_task(0, make_loader())
    tr._start_task(1)

    in_optimizer = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
    assert {id(p) for p in tr.backbone.heads["1"].parameters()} <= in_optimizer
    assert not {id(p) for p in tr.backbone.heads["0"].parameters()} & in_optimizer
    assert {id(p) for p in tr.backbone.backbone_parameters()} <= in_optimizer


def test_head_group_uses_scaled_learning_rate():
    tr = make_trainer(lr=1e-3, head_lr_scale=50.0)
    tr._start_task(0)
    lrs = sorted(g["lr"] for g in tr.optimizer.param_groups)
    assert lrs == pytest.approx([1e-3, 5e-2])


def test_new_head_trains_and_old_head_is_untouched():
    tr = make_trainer()
    tr.train_task(0, make_loader())
    old = {k: v.clone() for k, v in tr.backbone.heads["0"].state_dict().items()}

    tr._start_task(1)
    new_before = tr.backbone.heads["1"].weight.clone()
    tr._run_warmup(1, make_loader())

    assert not torch.allclose(tr.backbone.heads["1"].weight, new_before), "new head never trained"
    for k, v in tr.backbone.heads["0"].state_dict().items():
        assert torch.equal(v, old[k]), f"old head parameter {k} changed"


def test_frozen_backbone_parameters_do_not_change_and_trainable_ones_do():
    tr = make_trainer()
    before = {n: p.detach().clone() for n, p in tr.backbone.model.named_parameters()}

    tr.train_task(0, make_loader())

    for n, p in tr.backbone.model.named_parameters():
        changed = not torch.equal(p, before[n])
        if p.requires_grad:
            assert changed, f"trainable {n} never updated"
        else:
            assert not changed, f"frozen {n} changed"


def test_tasks_must_be_trained_in_order():
    tr = make_trainer()
    with pytest.raises(ValueError, match="gap"):
        tr.train_task(1, make_loader())


def test_a_task_can_be_trained_again():
    tr = make_trainer()
    tr.train_task(0, make_loader())
    tr.train_task(0, make_loader())
    assert tr.backbone.n_tasks == 1


# ── evaluation ───────────────────────────────────────────────────────────────

def test_evaluate_uses_the_head_of_the_requested_task():
    tr = make_trainer()
    run_tasks_until(tr, 2)
    with torch.no_grad():
        for tid, cls in ((0, 2), (1, 4)):
            head = tr.backbone.heads[str(tid)]
            head.weight.zero_()
            head.bias.zero_()
            head.bias[cls] = 10.0

    only_class_2 = make_loader(labels=2)
    assert tr.evaluate(0, only_class_2) == 1.0
    assert tr.evaluate(1, only_class_2) == 0.0
    assert tr.evaluate(1, make_loader(labels=4)) == 1.0


def test_evaluate_restores_training_mode():
    tr = make_trainer()
    tr.train_task(0, make_loader())
    tr.backbone.train()
    tr.evaluate(0, make_loader())
    assert tr.backbone.training
