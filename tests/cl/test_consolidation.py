"""
Tests for hippocortex.cl.consolidation (Phase 3, the SWR sleep phase).

OWNER: Hasinthaka Piyumal

CPU only, with the real SWRGenerator / StatsBuffer / NullSpaceProjector on small widths.
"""
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hippocortex.cl.consolidation import collect_hidden_states, consolidate
from hippocortex.cl.null_space_projector import NullSpaceProjector
from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.models.swr_generator import SWRGenerator

D = 16
N_TASKS = 4


def make_parts(seed: int = 0):
    torch.manual_seed(seed)
    gen = SWRGenerator(hidden_dim=D, latent_dim=4, n_tasks=N_TASKS)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    return gen, NullSpaceProjector(rank_budget=D), StatsBuffer(), opt


def hidden(task_id: int, n: int = 64, rank: int = 3) -> torch.Tensor:
    """Low-rank task features with a task-specific offset, so U leaves a null space."""
    g = torch.Generator().manual_seed(10 + task_id)
    return torch.randn(n, rank, generator=g) @ torch.randn(rank, D, generator=g) + 2.0 * task_id


BETA = 1e-3


def elbo(gen: SWRGenerator, h: torch.Tensor, task_id: int, draws: int = 20) -> float:
    """ELBO with the training beta, averaged over draws of the reparameterisation noise."""
    torch.manual_seed(123)
    total = 0.0
    with torch.no_grad():
        for _ in range(draws):
            recon, mu_q, logvar_q = gen(h, task_id)
            total += SWRGenerator.elbo_loss(recon, h, mu_q, logvar_q, beta=BETA).item()
    return total / draws


def run(parts, task_id: int, h: torch.Tensor, **kw) -> float:
    gen, proj, buf, opt = parts
    kw = {"n_samples": 32, "epochs": 1, "batch_size": 16, "beta": BETA, **kw}
    return consolidate(gen, proj, buf, task_id, opt, h, **kw)


# ── bookkeeping ──────────────────────────────────────────────────────────────

def test_stores_the_stats_of_the_real_hidden_states():
    parts = make_parts()
    h = hidden(0)
    run(parts, 0, h, epochs=0)

    mu, sigma2 = parts[2].get_stats(0)
    assert torch.allclose(mu, h.mean(0), atol=1e-6)
    assert torch.allclose(sigma2, h.var(0, unbiased=False), atol=1e-5)


def test_extends_the_null_space_basis_to_cover_the_task():
    parts = make_parts()
    proj = parts[1]
    h0, h1 = hidden(0), hidden(1)

    run(parts, 0, h0, epochs=0)
    rank0 = proj.current_rank
    assert 0 < rank0 < D
    U = proj._U
    assert (h0 - h0 @ U @ U.T).norm() < 0.1 * h0.norm()  # >= 99% of the energy is protected

    run(parts, 1, h1, epochs=0)
    assert proj.current_rank > rank0


def test_zero_epochs_only_does_the_bookkeeping():
    parts = make_parts()
    before = [p.clone() for p in parts[0].parameters()]
    assert run(parts, 0, hidden(0), epochs=0) == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, parts[0].parameters()))
    assert parts[2].task_ids() == [0]


def test_rejects_hidden_states_of_the_wrong_width():
    with pytest.raises(ValueError, match="must be"):
        run(make_parts(), 0, torch.randn(8, D + 1))


# ── CVAE fine-tune ───────────────────────────────────────────────────────────

def test_fine_tune_lowers_the_elbo_on_the_current_task():
    parts = make_parts()
    h = hidden(0)
    before = elbo(parts[0], h, 0)
    loss = run(parts, 0, h, epochs=60)
    assert loss > 0
    assert elbo(parts[0], h, 0) < 0.5 * before


def test_task0_has_nothing_to_replay():
    parts = make_parts()
    parts[0].sample = MagicMock(wraps=parts[0].sample)
    run(parts, 0, hidden(0))
    parts[0].sample.assert_not_called()


def test_each_earlier_task_is_replayed_once_before_any_step():
    parts = make_parts()
    gen, _, buf, opt = parts
    run(parts, 0, hidden(0), epochs=0)
    run(parts, 1, hidden(1), epochs=0)

    steps = {"n": 0}
    real_step = opt.step

    def counting_step(*a, **k):
        steps["n"] += 1
        return real_step(*a, **k)

    opt.step = counting_step
    calls = []
    real_sample = gen.sample

    def spy(task_id, n, stats, device=None):
        calls.append((task_id, n, steps["n"]))
        return real_sample(task_id, n, stats, device)

    gen.sample = spy
    run(parts, 2, hidden(2), n_samples=24)

    assert [(t, n) for t, n, _ in calls] == [(0, 24), (1, 24)]
    assert all(s == 0 for _, _, s in calls)  # sampled from the generator as it was before sleep
    assert steps["n"] > 0


def test_replayed_tasks_and_the_current_task_are_trained_others_are_not():
    parts = make_parts()
    gen = parts[0]
    run(parts, 0, hidden(0), epochs=0)
    emb_before = gen.task_embedding.weight.detach().clone()

    run(parts, 1, hidden(1), epochs=2)

    changed = (gen.task_embedding.weight.detach() - emb_before).abs().sum(dim=1) > 0
    assert changed[0] and changed[1]          # task 1 (current) and task 0 (replayed)
    assert not changed[2] and not changed[3]  # tasks not seen yet


def test_replay_keeps_the_old_task_better_than_no_replay():
    """Fine-tuning on task 1 alone makes the generator forget task 0; replay limits it."""
    h0, h1 = hidden(0), hidden(1)

    def forgetting(replay: bool) -> float:
        parts = make_parts(seed=1)
        run(parts, 0, h0, epochs=80)
        before = elbo(parts[0], h0, 0)
        if not replay:
            parts[2]._store.clear()  # nothing to replay
        run(parts, 1, h1, epochs=80)
        return elbo(parts[0], h0, 0) - before

    assert forgetting(replay=True) < forgetting(replay=False)


# ── collect_hidden_states ────────────────────────────────────────────────────

class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, D)
        self.drop = nn.Dropout(0.5)

    def extract_hidden(self, x):
        return self.drop(self.lin(x))


def test_collect_hidden_states_runs_in_eval_mode_without_grad_and_restores_mode():
    enc = _Encoder().train()
    x = torch.randn(10, 3)
    loader = DataLoader(TensorDataset(x, torch.zeros(10)), batch_size=4)

    h = collect_hidden_states(enc, loader, torch.device("cpu"))

    assert h.shape == (10, D) and h.dtype == torch.float32 and not h.requires_grad
    assert torch.allclose(h, enc.lin(x).detach(), atol=1e-6)  # dropout off: eval mode
    assert enc.training


def test_collect_hidden_states_rejects_an_empty_loader():
    loader = DataLoader(TensorDataset(torch.empty(0, 3), torch.empty(0)), batch_size=4)
    with pytest.raises(ValueError, match="empty"):
        collect_hidden_states(_Encoder(), loader, torch.device("cpu"))
