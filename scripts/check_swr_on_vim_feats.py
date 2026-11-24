"""
Check the SWR generator (CVAE) and the StatsBuffer on real Vim-small features, on CPU.

Usage:
    python scripts/check_swr_on_vim_feats.py --feats out/vim_feats --tasks 0 1 2

--feats is the folder written by scripts/extract_vim_feats.py (train.pt, test.pt).

Steps:
1. StatsBuffer: update() per task, check get_stats() against feats.mean(0) / feats.var(0)
   (population variance, as StatsBuffer stores it), report memory_bytes().
2. Train the CVAE on the --train-task's train features (the trainer's recipe: Adam, lr 3e-4,
   beta = kl_weight 1e-3) and log the ELBO per epoch.
3. Sample features for that task and measure how useful they are:
   - real:          logistic regression trained on real train features, tested on real test
                    features (the upper bound);
   - cvae:          StatsBuffer stats -> SWRGenerator.sample (the trainer's replay path);
   - rough:         the Gaussian draw from (mu, sigma^2) alone, without the CVAE;
   - per_class:     the same CVAE recipe, but with per-class stats and the class as the
                    condition (memory O(classes x d) instead of O(tasks x d)).
   sample() is per task, so its samples have no class label. "cvae" and "rough" samples are
   labelled by the real-feature classifier (as replay through a shared head would use the old
   head's logits); a new classifier is trained on them and tested on the real test features.
   "per_class" samples carry their class.
   Each row also reports how close the samples are to the real train features: the mean
   absolute error of the per-dimension mean, and the mean ratio of per-dimension std.

Writes <feats>/swr_check.json and prints a summary. A gap between "real" and "cvae" is a
finding, not a failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hippocortex.cl.stats_buffer import StatsBuffer  # noqa: E402
from hippocortex.models.swr_generator import SWRGenerator  # noqa: E402


def load(feats_dir: Path) -> tuple[dict, dict]:
    train = torch.load(feats_dir / "train.pt", weights_only=True)
    test = torch.load(feats_dir / "test.pt", weights_only=True)
    for d in (train, test):
        d["feats"] = d["feats"].float()
    return train, test


def select(d: dict, task: int) -> tuple[torch.Tensor, torch.Tensor]:
    m = d["task_ids"] == task
    return d["feats"][m], d["labels"][m]


def check_stats(train: dict, tasks: list[int]) -> dict:
    buf = StatsBuffer()
    worst = 0.0
    for t in tasks:
        f, _ = select(train, t)
        buf.update(t, f)
        mu, var = buf.get_stats(t)
        worst = max(worst, float((mu - f.mean(0)).abs().max()),
                    float((var - f.var(0, unbiased=False)).abs().max()))
    d = train["feats"].shape[1]
    mem = buf.memory_bytes()
    ok = worst < 1e-5 and mem == len(tasks) * d * 2 * 4
    return {"ok": ok, "max_abs_err": worst, "memory_bytes": mem, "tasks": tasks, "buffer": buf}


def train_cvae(feats: torch.Tensor, cond: torch.Tensor, n_cond: int, epochs: int, latent_dim: int,
               lr: float, beta: float, batch_size: int, seed: int) -> tuple[SWRGenerator, list[float]]:
    """Train an SWRGenerator on feats, conditioned on cond (task or class ids)."""
    torch.manual_seed(seed)
    gen = SWRGenerator(hidden_dim=feats.shape[1], latent_dim=latent_dim, n_tasks=n_cond)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    curve = []
    for _ in range(epochs):
        perm = torch.randperm(len(feats))
        total, n = 0.0, 0
        for c in cond.unique().tolist():
            idx = perm[cond[perm] == c]
            for i in range(0, len(idx), batch_size):
                h = feats[idx[i : i + batch_size]]
                recon, mu_q, logvar_q = gen(h, int(c))
                loss = SWRGenerator.elbo_loss(recon, h, mu_q, logvar_q, beta=beta)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(h)
                n += len(h)
        curve.append(total / n)
    return gen.eval(), curve


def probe(x_tr, y_tr, x_te, y_te):
    """Standardised logistic regression; returns (accuracy, fitted (scaler, clf))."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=3000).fit(scaler.transform(x_tr), y_tr)
    return float(clf.score(scaler.transform(x_te), y_te)), (scaler, clf)


def closeness(samples: torch.Tensor, real: torch.Tensor) -> dict:
    return {
        "mean_abs_err": round(float((samples.mean(0) - real.mean(0)).abs().mean()), 4),
        "std_ratio": round(float((samples.std(0) / real.std(0).clamp_min(1e-6)).mean()), 4),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", default="out/vim_feats")
    ap.add_argument("--tasks", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--train-task", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=None, help="default: as many as real train features")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    feats_dir = Path(args.feats)
    train, test = load(feats_dir)
    present = set(train["task_ids"].unique().tolist())
    tasks = [t for t in args.tasks if t in present]
    if args.train_task not in present:
        raise SystemExit(f"task {args.train_task} is not in {feats_dir / 'train.pt'} (has {sorted(present)})")

    # 1. stats buffer
    stats = check_stats(train, tasks)
    buf = stats.pop("buffer")
    print(f":: stats check {'OK' if stats['ok'] else 'FAILED'} (max abs err {stats['max_abs_err']:.2e}), "
          f"memory {stats['memory_bytes']} bytes for {len(tasks)} tasks x {train['feats'].shape[1]} dims")

    t = args.train_task
    x_tr, y_tr = select(train, t)
    x_te, y_te = select(test, t)
    n = args.n_samples or len(x_tr)
    common = dict(epochs=args.epochs, latent_dim=args.latent_dim, lr=args.lr, beta=args.beta,
                  batch_size=args.batch_size, seed=args.seed)

    # 2. CVAE on the task's features, conditioned on the task (the trainer's setup)
    gen, curve = train_cvae(x_tr, torch.full((len(x_tr),), t), n_cond=t + 1, **common)
    print(":: ELBO per epoch: " + " ".join(f"{v:.4f}" for v in curve))

    # 3. how useful are the samples?
    acc_real, (scaler, teacher) = probe(x_tr.numpy(), y_tr.numpy(), x_te.numpy(), y_te.numpy())
    torch.manual_seed(args.seed)
    with torch.no_grad():
        s_cvae = gen.sample(t, n, buf.get_stats(t), torch.device("cpu"))
    mu, var = buf.get_stats(t)
    s_rough = mu + var.clamp_min(0).sqrt() * torch.randn(n, len(mu))

    rows = {"real": {"acc": round(acc_real, 4), **closeness(x_tr, x_tr)}}
    for name, s in (("cvae", s_cvae), ("rough", s_rough)):
        pseudo = teacher.predict(scaler.transform(s.numpy()))
        if len(np.unique(pseudo)) < 2:
            acc = float("nan")  # every sample got one label: no classifier can be trained
        else:
            acc, _ = probe(s.numpy(), pseudo, x_te.numpy(), y_te.numpy())
        rows[name] = {"acc": round(acc, 4), "pseudo_label_classes": int(len(np.unique(pseudo))),
                      **closeness(s, x_tr)}

    # per-class variant: class stats + class-conditioned CVAE
    classes = sorted(y_tr.unique().tolist())
    cls_idx = torch.tensor([classes.index(int(c)) for c in y_tr])
    gen_c, curve_c = train_cvae(x_tr, cls_idx, n_cond=len(classes), **common)
    cls_buf = StatsBuffer()
    per_cls, ys = [], []
    torch.manual_seed(args.seed)
    for i, c in enumerate(classes):
        cls_buf.update(i, x_tr[cls_idx == i])
        k = max(2, n // len(classes))
        with torch.no_grad():
            per_cls.append(gen_c.sample(i, k, cls_buf.get_stats(i), torch.device("cpu")))
        ys += [c] * k
    s_cls = torch.cat(per_cls)
    acc_cls, _ = probe(s_cls.numpy(), np.array(ys), x_te.numpy(), y_te.numpy())
    rows["per_class"] = {"acc": round(acc_cls, 4), "memory_bytes": cls_buf.memory_bytes(),
                         **closeness(s_cls, x_tr)}

    result = {
        "feats": str(feats_dir),
        "n_train": len(x_tr), "n_test": len(x_te), "n_samples": n, "train_task": t,
        "stats_check": stats,
        "elbo_curve": [round(v, 5) for v in curve],
        "elbo_curve_per_class": [round(v, 5) for v in curve_c],
        "elbo_decreased": curve[-1] < curve[0],
        "linear_probe": rows,
        "settings": vars(args),
    }
    (feats_dir / "swr_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f":: task {t}: {len(x_tr)} train / {len(x_te)} test real features, {n} samples")
    print(f"   {'features':<10} {'probe acc':>9} {'mean err':>9} {'std ratio':>9}")
    for name, r in rows.items():
        print(f"   {name:<10} {r['acc']:>9.2%} {r['mean_abs_err']:>9.4f} {r['std_ratio']:>9.3f}")
    return result


if __name__ == "__main__":
    main()
