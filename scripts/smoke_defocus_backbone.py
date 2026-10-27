"""
Smoke test for the De-focus Mamba base backbone and the Phase 2 joint loop.

Run on the GPU machine (Kaggle T4) after the environment from
notebooks/kaggle/*_mamba_cl_base_backbone.ipynb is installed (mamba-ssm 1.2.2,
causal-conv1d 1.4.0, `pip install -e .`):

    wget -O checkpoints/defocus_mamba_base_cls.pth \\
      https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/defocus_mamba_base_cls.pth
    python scripts/smoke_defocus_backbone.py                    # batch 8
    python scripts/smoke_defocus_backbone.py --batch-size 64    # the real batch size: memory check

What it checks (each line prints PASS / FAIL; exit code 1 if any FAIL):
  1. The checkpoint loads: no `blocks.*` / `patch_embed.*` keys missing (key report is printed).
  2. forward(images) -> logits (B, 5), hidden (B, 768); 32x32 inputs are resized to 224.
  3. Only head / x_proj / out_proj.weight / A_log are trainable.
  4. Two tasks through Trainer.train_task (warmup + joint, fp16 AMP on CUDA): losses are finite,
     the old head is untouched, the new head and backbone parameters train, frozen ones do not.
  5. Phase 2 replay and projection actually ran on task 1 (the consolidation phase is
     simulated here, because consolidate() is not implemented yet).
  6. Peak GPU memory and seconds per step, to size the real run.

`--cpu-dry-run` swaps in the tiny CPU stand-in so the script itself can be tested anywhere.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from hippocortex.cl.null_space_projector import NullSpaceProjector  # noqa: E402
from hippocortex.cl.stats_buffer import StatsBuffer  # noqa: E402
from hippocortex.models.backbone import BACKBONE_TRAINABLE_PATTERNS, MambaBackbone  # noqa: E402
from hippocortex.models.fake_defocus import TinyDefocus  # noqa: E402
from hippocortex.models.swr_generator import SWRGenerator  # noqa: E402
from hippocortex.training.trainer import Trainer  # noqa: E402
from hippocortex.utils.config import load_config  # noqa: E402

CONFIG = REPO_ROOT / "experiments" / "stage1_split_cifar100" / "config.yaml"
N_CLASSES = 5
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


def make_loader(n: int, batch_size: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, N_CLASSES, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)


def snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Parameter copy on the CPU, so it does not add to the GPU memory being measured."""
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters()}


def changed(before: dict[str, torch.Tensor], module: torch.nn.Module, names) -> list[str]:
    now = dict(module.named_parameters())
    return [n for n in names if not torch.equal(before[n], now[n].detach().cpu())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "defocus_mamba_base_cls.pth")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cpu-dry-run", action="store_true", help="use the tiny CPU stand-in instead of the real model")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} | cuda={cuda}" + (f" | {torch.cuda.get_device_name(0)}" if cuda else ""))
    if not args.cpu_dry_run and not cuda:
        print("The real model needs a CUDA GPU (use --cpu-dry-run to test this script on CPU).")
        return 2

    # ── 1. build + load ────────────────────────────────────────────────
    t0 = time.time()
    if args.cpu_dry_run:
        # d_inner != embed_dim, as in the real model (1536 vs 768); equal sizes would make
        # the shape-based projection treat every weight as projectable.
        backbone = MambaBackbone(model=TinyDefocus(embed_dim=64, depth=2, d_inner=96), image_size=64)
    else:
        backbone = MambaBackbone(checkpoint_path=args.checkpoint)  # raises if blocks.* / patch_embed.* missing
    check("checkpoint loaded, backbone weights present", True, f"{time.time() - t0:.1f}s")
    dim = backbone.feature_dim
    if not args.cpu_dry_run:
        check("feature_dim == 768", dim == 768, f"got {dim}")

    cfg = load_config(CONFIG)
    cfg.training.epochs_warmup = 1
    cfg.training.epochs_joint = 1
    cfg.dataset.n_classes_per_task = N_CLASSES
    device = torch.device("cuda" if cuda else "cpu")
    backbone.to(device)

    # ── 2. forward ─────────────────────────────────────────────────────
    backbone.set_task_head(N_CLASSES, task_id=0)
    backbone.eval()
    with torch.no_grad():
        logits, hidden = backbone(torch.randn(2, 3, 32, 32, device=device))
    check("forward: logits (2, 5), hidden (2, feature_dim)", logits.shape == (2, N_CLASSES) and hidden.shape == (2, dim),
          f"logits {tuple(logits.shape)}, hidden {tuple(hidden.shape)}")
    check("forward: outputs finite", bool(torch.isfinite(logits).all() and torch.isfinite(hidden).all()))
    backbone.heads.clear()  # trainer creates head 0 itself

    # ── 3. trainable scope ─────────────────────────────────────────────
    trainable = [n for n, p in backbone.model.named_parameters() if p.requires_grad]
    stray = [n for n in trainable if not any(pat in n for pat in BACKBONE_TRAINABLE_PATTERNS)]
    check("only x_proj / out_proj.weight / A_log trainable in the backbone", bool(trainable) and not stray,
          f"{len(trainable)} tensors, stray={stray[:3]}")

    # ── 4/5. two tasks through the Trainer ─────────────────────────────
    swr = SWRGenerator(hidden_dim=dim, latent_dim=cfg.model.cvae_latent_dim, n_tasks=cfg.dataset.n_tasks)
    buffer, projector = StatsBuffer(), NullSpaceProjector(rank_budget=cfg.model.nsp_rank_budget)
    trainer = Trainer(backbone, swr, projector, buffer, cfg)

    initial = snapshot(backbone)  # before any training; fp16 GradScaler may skip a few early steps
    initial_heads: dict[int, dict[str, torch.Tensor]] = {}
    create_head = backbone.set_task_head

    def spy_set_task_head(n_classes, task_id=None):
        head = create_head(n_classes, task_id=task_id)
        initial_heads[task_id] = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        return head

    backbone.set_task_head = spy_set_task_head
    bs = args.batch_size
    n = 4 * bs

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    m0 = trainer.train_task(0, make_loader(n, bs, seed=0))
    dt0 = time.time() - t0
    check("task 0: losses finite", all(map(lambda v: v == v and abs(v) != float("inf"), m0.values())), str({k: round(v, 3) for k, v in m0.items()}))
    check("task 0: no replay, no projection", m0["replay_loss"] == 0.0 and m0["proj_params_projected"] == 0)

    # Stand-in for Phase 3 (consolidate() is not implemented yet): store task-0 statistics + basis.
    backbone.eval()
    with torch.no_grad():
        feats = torch.cat([backbone.extract_hidden(x.to(device)).float() for x, _ in make_loader(n, bs, seed=0)])
    backbone.train()
    buffer.update(0, feats)
    projector.update(feats)
    print(f"      simulated consolidation: buffer tasks={buffer.task_ids()}, projector rank={projector.current_rank}")

    head0_before = {k: v.detach().cpu().clone() for k, v in backbone.heads["0"].state_dict().items()}
    t0 = time.time()
    m1 = trainer.train_task(1, make_loader(n, bs, seed=1))
    dt1 = time.time() - t0
    check("task 1: losses finite", all(map(lambda v: v == v and abs(v) != float("inf"), m1.values())), str({k: round(v, 3) for k, v in m1.items()}))
    check("task 1: replay ran (past task sampled)", m1["replay_loss"] > 0.0, f"replay={m1['replay_loss']:.4f}")
    expected_projected = 1 + len(backbone.model.blocks)  # head weight + one out_proj per block
    check("task 1: head weight + every out_proj projected", m1["proj_params_projected"] == expected_projected,
          f"projected={int(m1['proj_params_projected'])} (expected {expected_projected}), skipped={int(m1['proj_params_skipped'])}")
    check("task 1: old head untouched",
          all(torch.equal(v, backbone.heads["0"].state_dict()[k].cpu()) for k, v in head0_before.items()))
    check("task 1: new head trained",
          any(not torch.equal(v, backbone.heads["1"].state_dict()[k].cpu()) for k, v in initial_heads[1].items()))
    frozen = [n for n, p in backbone.named_parameters() if not p.requires_grad and n.startswith("model.")]
    check("both tasks: frozen backbone parameters unchanged", not changed(initial, backbone, frozen),
          f"{len(frozen)} frozen tensors")
    train_names = [n for n, p in backbone.named_parameters() if p.requires_grad and n.startswith("model.")]
    moved = changed(initial, backbone, train_names)
    check("both tasks: trainable backbone parameters moved", len(moved) > 0,
          f"{len(moved)}/{len(train_names)} tensors; GradScaler scale={trainer.scaler.get_scale() if trainer.use_amp else 'off'}")

    acc0, acc1 = trainer.evaluate(0, make_loader(n, bs, seed=2)), trainer.evaluate(1, make_loader(n, bs, seed=3))
    check("evaluate: accuracies in [0, 1] (random data, so ~0.2 is expected)", 0 <= acc0 <= 1 and 0 <= acc1 <= 1,
          f"task0={acc0:.2f} task1={acc1:.2f}")

    # ── 6. cost ────────────────────────────────────────────────────────
    steps = 2 * (cfg.training.epochs_warmup + cfg.training.epochs_joint) * (n // bs)
    print(f"\nbatch={bs} | {steps} train steps in {dt0 + dt1:.1f}s = {(dt0 + dt1) / steps:.2f}s/step"
          + (f" | peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if cuda else ""))

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
