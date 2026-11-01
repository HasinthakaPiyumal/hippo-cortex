"""
Tests for Phase 3 (consolidation) inside Trainer.train_task.

OWNER: Hasinthaka Piyumal

CPU only, with a TinyDefocus stand-in backbone and the real SWRGenerator / StatsBuffer /
NullSpaceProjector.
"""
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from hippocortex.cl.null_space_projector import NullSpaceProjector
from hippocortex.cl.stats_buffer import StatsBuffer
from hippocortex.models.backbone import MambaBackbone
from hippocortex.models.fake_defocus import TinyDefocus
from hippocortex.models.swr_generator import SWRGenerator
from hippocortex.training.trainer import Trainer

D = 32
N_CLS = 5


def make_trainer(**training) -> Trainer:
    t = dict(
        lr=1e-2, head_lr_scale=10.0, weight_decay=0.0, betas=[0.9, 0.999],
        epochs_warmup=1, epochs_joint=1, epochs_consolidation=2, grad_clip=1.0, kl_weight=1e-3,
        use_amp=False, cvae_lr=1e-3, cvae_batch_size=8, replay_samples=16,
    )
    t.update(training)
    cfg = OmegaConf.create({"dataset": {"n_classes_per_task": N_CLS}, "training": t})
    torch.manual_seed(0)
    backbone = MambaBackbone(model=TinyDefocus(embed_dim=D, depth=2), image_size=64)
    swr = SWRGenerator(hidden_dim=D, latent_dim=8, n_tasks=4)
    return Trainer(backbone, swr, NullSpaceProjector(rank_budget=16), StatsBuffer(), cfg)


def make_loader(seed: int = 0, n: int = 16) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, N_CLS, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_train_task_consolidates_the_task():
    tr = make_trainer()
    gen_before = [p.clone() for p in tr.swr_gen.parameters()]

    metrics = tr.train_task(0, make_loader())

    assert tr.buffer.task_ids() == [0]
    assert tr.projector.current_rank > 0
    assert metrics["loss_consolidation"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(gen_before, tr.swr_gen.parameters()))


def test_stored_stats_are_the_eval_mode_features_after_training():
    tr = make_trainer()
    loader = make_loader()
    tr.train_task(0, loader)

    tr.backbone.eval()
    with torch.no_grad():
        h = torch.cat([tr.backbone.extract_hidden(x) for x, _ in loader])
    mu, sigma2 = tr.buffer.get_stats(0)
    assert torch.allclose(mu, h.mean(0), atol=1e-5)
    assert torch.allclose(sigma2, h.var(0, unbiased=False), atol=1e-5)


def test_the_next_task_replays_and_projects_after_consolidation():
    tr = make_trainer()
    tr.train_task(0, make_loader(seed=0))

    metrics = tr.train_task(1, make_loader(seed=1))

    assert metrics["replay_loss"] > 0             # Phase 2 sees task 0 in the buffer
    assert metrics["proj_params_projected"] > 0   # and has a basis to project with
    assert tr.buffer.task_ids() == [0, 1]


def test_consolidate_false_skips_phase_3():
    tr = make_trainer(consolidate=False)
    metrics = tr.train_task(0, make_loader())
    assert tr.buffer.task_ids() == []
    assert tr.projector.current_rank == 0
    assert metrics["loss_consolidation"] == 0.0


def test_zero_consolidation_epochs_still_does_the_bookkeeping():
    tr = make_trainer(epochs_consolidation=0)
    gen_before = [p.clone() for p in tr.swr_gen.parameters()]
    metrics = tr.train_task(0, make_loader())
    assert tr.buffer.task_ids() == [0] and tr.projector.current_rank > 0
    assert metrics["loss_consolidation"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(gen_before, tr.swr_gen.parameters()))


def test_the_generator_optimizer_is_kept_across_tasks():
    tr = make_trainer()
    opt = tr.swr_optimizer
    tr.train_task(0, make_loader(seed=0))
    tr.train_task(1, make_loader(seed=1))
    assert tr.swr_optimizer is opt
    assert len(opt.state) > 0
