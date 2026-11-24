"""
Extract Vim-small features of Split-CIFAR100 once, for CPU work on the SWR generator, the
StatsBuffer and the null-space projector.

Usage (Kaggle GPU, ~10 min on a T4):
    python scripts/extract_vim_feats.py --data-root data/cifar100 --out /kaggle/working/vim_feats

Quick CPU check on a few images per task (Vim-small runs ~1 img/s on CPU):
    python scripts/extract_vim_feats.py --tasks 0 --limit 64 --device cpu --out out/vim_feats_small

What it does:
- Benchmark as our Inf-SSM run: CIFAR-100, 10 tasks of 10 classes in natural order
  (task t = classes 10t..10t+9), seed 0.
- Model: Vim-small with the authors' task-0 checkpoint (HF ileening0022/Inf-SSM,
  cifar100_10task/task_0_checkpoint_seed_0.pth), loaded through MambaBackbone, which
  checks that no backbone key is missing.
- Features: MambaBackbone.extract_hidden = encode_image(x, pre_logits=True), the middle class
  token after the final norm, (B, 384). eval(), no_grad, fp16 autocast on CUDA.
- Transform: the Inf-SSM eval transform for train and test images alike (resize 256 + centre
  crop 224, CIFAR-100 stats), with no augmentation.

Output in --out:
    train.pt, test.pt   {"feats": float16 (N, 384), "labels": int64 global ids (N,),
                         "task_ids": int64 (N,)}
    meta.json           checkpoint, seed, class order, transform, counts, date
    README.md           sanity numbers: counts, per-task feature mean/std, and the accuracy of
                        a logistic regression trained on one task's train features and tested
                        on its test features (task 0 and the last task extracted)

These are features of a FROZEN task-0 backbone, not of a backbone trained continually. They
are fine for developing the SWR generator and the stats buffer; they are not a CL result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hippocortex.data.split_cifar100 import _build_transform, download_if_missing  # noqa: E402
from hippocortex.models.backbone import MambaBackbone, task0_checkpoint  # noqa: E402

N_TASKS = 10
PER_TASK = 10
HF_DIR = "cifar100_10task"
SEED = 0


def task_indices(targets: list[int], task: int, limit: int | None) -> list[int]:
    classes = set(range(task * PER_TASK, (task + 1) * PER_TASK))
    idx = [i for i, y in enumerate(targets) if y in classes]
    if limit is not None:
        # Spread the limit over the classes so every class is represented.
        per_class = max(1, limit // PER_TASK)
        by_class: dict[int, list[int]] = {}
        for i in idx:
            by_class.setdefault(targets[i], []).append(i)
        idx = sorted(i for c in sorted(by_class) for i in by_class[c][:per_class])
    return idx


@torch.no_grad()
def extract(backbone: MambaBackbone, dataset, tasks: list[int], limit: int | None,
            batch_size: int, num_workers: int, device: torch.device) -> dict[str, torch.Tensor]:
    feats, labels, task_ids = [], [], []
    for t in tasks:
        sub = Subset(dataset, task_indices(dataset.targets, t, limit))
        loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=device.type == "cuda")
        start = time.time()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                h = backbone.extract_hidden(x)
            feats.append(h.float().cpu().half())
            labels.append(y.long())
            task_ids.append(torch.full_like(y, t, dtype=torch.long))
        print(f"  task {t}: {len(sub)} images in {time.time() - start:.0f}s", flush=True)
    return {"feats": torch.cat(feats), "labels": torch.cat(labels), "task_ids": torch.cat(task_ids)}


def linear_probe(train: dict, test: dict, task: int) -> float:
    """Logistic regression on one task's train features, accuracy on its test features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    tr, te = train["task_ids"] == task, test["task_ids"] == task
    x_tr, y_tr = train["feats"][tr].float().numpy(), train["labels"][tr].numpy()
    x_te, y_te = test["feats"][te].float().numpy(), test["labels"][te].numpy()
    if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
        return float("nan")
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(x_tr), y_tr)
    return float(clf.score(scaler.transform(x_te), y_te))


def sanity(train: dict, test: dict, tasks: list[int]) -> dict:
    stats = {"counts": {"train": len(train["labels"]), "test": len(test["labels"])}, "per_task": {}}
    for t in tasks:
        f = train["feats"][train["task_ids"] == t].float()
        stats["per_task"][t] = {
            "n_train": int((train["task_ids"] == t).sum()),
            "n_test": int((test["task_ids"] == t).sum()),
            "feat_mean": round(float(f.mean()), 4),
            "feat_std": round(float(f.std()), 4),
            "feat_norm": round(float(f.norm(dim=1).mean()), 3),
        }
    probe_tasks = sorted({tasks[0], tasks[-1]})
    stats["linear_probe_acc"] = {t: round(linear_probe(train, test, t), 4) for t in probe_tasks}
    return stats


def write_readme(out: Path, meta: dict, stats: dict) -> None:
    lines = [
        "# Vim-small features of Split-CIFAR100",
        "",
        "Features of the **frozen** task-0 Vim-small backbone (authors' checkpoint), not of a",
        "backbone trained continually. For developing the SWR generator and the stats buffer on CPU;",
        "not a continual-learning result.",
        "",
        f"- Checkpoint: `{meta['checkpoint']}`",
        f"- Benchmark: CIFAR-100, {N_TASKS} tasks x {PER_TASK} classes, natural order, seed {SEED}",
        f"- Transform: {meta['transform']}",
        f"- Feature: {meta['feature']}",
        f"- Files: `train.pt`, `test.pt` with `feats` float16 (N, {meta['feature_dim']}), `labels` "
        "(global class ids), `task_ids`",
        f"- Images: {stats['counts']['train']} train, {stats['counts']['test']} test"
        + (f" (limit {meta['limit']} per task per split)" if meta["limit"] else ""),
        "",
        "## Sanity checks",
        "",
        "| task | train | test | feat mean | feat std | mean L2 norm |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for t, s in stats["per_task"].items():
        lines.append(f"| {t} | {s['n_train']} | {s['n_test']} | {s['feat_mean']} | {s['feat_std']} | {s['feat_norm']} |")
    lines += ["", "Logistic regression trained on a task's train features, tested on its test features:", ""]
    for t, acc in stats["linear_probe_acc"].items():
        lines.append(f"- task {t}: {acc:.2%}")
    lines += ["", "Task 0 should be high: the checkpoint was fine-tuned on task 0.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/cifar100")
    ap.add_argument("--out", default="out/vim_feats")
    ap.add_argument("--tasks", type=int, nargs="*", default=list(range(N_TASKS)))
    ap.add_argument("--limit", type=int, default=None, help="images per task per split (quick checks)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", default="hf", help="'hf' (task-0 checkpoint) or a local path")
    args = ap.parse_args(argv)

    from torchvision import datasets

    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    download_if_missing(args.data_root)
    tf = _build_transform("inf_ssm", train=False)  # resize 256 + centre crop 224, CIFAR stats
    data = {s: datasets.CIFAR100(args.data_root, train=s == "train", download=False, transform=tf)
            for s in ("train", "test")}

    ckpt = task0_checkpoint(HF_DIR, SEED) if args.checkpoint == "hf" else args.checkpoint
    backbone = MambaBackbone(checkpoint_path=ckpt, head_mode="shared", n_classes_total=100).to(device).eval()
    print(f":: extracting {backbone.feature_dim}-d features on {device} for tasks {args.tasks}", flush=True)

    feats = {}
    for split, ds in data.items():
        print(f":: {split}", flush=True)
        feats[split] = extract(backbone, ds, args.tasks, args.limit, args.batch_size, args.num_workers, device)
        torch.save(feats[split], out / f"{split}.pt")

    meta = {
        "checkpoint": f"ileening0022/Inf-SSM/{HF_DIR}/task_0_checkpoint_seed_{SEED}.pth"
        if args.checkpoint == "hf" else str(ckpt),
        "seed": SEED,
        "n_tasks": N_TASKS,
        "class_order": "natural (task t = classes 10t..10t+9)",
        "tasks": args.tasks,
        "limit": args.limit,
        "transform": "Inf-SSM eval: resize 256 bicubic, centre crop 224, CIFAR-100 mean/std",
        "feature": "MambaBackbone.extract_hidden = encode_image(x, pre_logits=True), middle cls token",
        "feature_dim": backbone.feature_dim,
        "dtype": "float16",
        "date": date.today().isoformat(),
    }
    stats = sanity(feats["train"], feats["test"], args.tasks)
    meta["sanity"] = stats
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_readme(out, meta, stats)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
