"""
HippoCortex Continual Learning: Mamba-CL Baseline Runner (3 Epochs).

Executes the Mamba-CL baseline on Class-Incremental benchmarks (CIFAR-100 / ImageNet-R)
for 3 epochs per task, using Null-Space gradient projection and Automatic Mixed Precision (AMP).

Usage:
    python scripts/mamba_cl/run_mamba_cl_3epochs.py
    python scripts/mamba_cl/run_mamba_cl_3epochs.py --batch_size 64
    python scripts/mamba_cl/run_mamba_cl_3epochs.py --tasks 2
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path


def find_project_root() -> Path:
    """Traverse up until project root containing baselines/mamba-cl is found."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "baselines" / "mamba-cl").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root directory containing 'baselines/mamba-cl'.")


def main():
    parser = argparse.ArgumentParser(description="Run Mamba-CL Baseline (3 Epochs per task)")
    parser.add_argument("-d", "--dataset", type=str, default="cifar100", choices=["cifar100", "imagenet_r"], help="Dataset name (default: cifar100)")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Epochs per task (default: 3)")
    parser.add_argument("-t", "--tasks", type=int, default=10, help="Number of continual learning tasks (default: 10)")
    parser.add_argument("-b", "--batch_size", type=int, default=200, help="Batch size (default: 200)")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed (default: 2024)")
    parser.add_argument("--data_root", type=str, default="", help="Path to formatted dataset images directory")
    parser.add_argument("--pretrained_path", type=str, default="", help="Path to defocus_mamba_large_cls_21k.pth")
    parser.add_argument("--use_amp", type=str, default="True", help="Enable Automatic Mixed Precision (True/False, default: True)")
    parser.add_argument("--workers", type=int, default=4 if os.name != 'nt' else 2, help="DataLoader workers (default: 2 on Windows, 4 on Linux)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")

    args, unknown_args = parser.parse_known_args()
    project_root = find_project_root()

    # 1. Resolve Pretrained Weights Path
    if not args.pretrained_path:
        pretrained_path = project_root / "defocus_mamba_large_cls_21k.pth"
    else:
        pretrained_path = Path(args.pretrained_path).resolve()

    if not pretrained_path.exists():
        print(f"\n[ERROR] Pretrained weights not found at: {pretrained_path}")
        print("Please download the weights using:")
        print("  wget https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/defocus_mamba_large_cls_21k.pth")
        print(f"and save to: {project_root / 'defocus_mamba_large_cls_21k.pth'}\n")
        sys.exit(1)

    # 2. Resolve Dataset Path
    if not args.data_root:
        if args.dataset == "cifar100":
            data_root = project_root / "data" / "cifar100-images"
        else:
            data_root = project_root / "data" / "imagenet-r"
    else:
        data_root = Path(args.data_root).resolve()

    if not data_root.exists() or not (data_root / "train").exists():
        print(f"\n[ERROR] Formatted dataset not found at: {data_root}")
        if args.dataset == "cifar100":
            print("Please download and prepare CIFAR-100 by running:")
            print("  1. python scripts/download_datasets.py --dataset cifar100")
            print("  2. python scripts/split_cifar100.py")
        elif args.dataset == "imagenet_r":
            print("Please download and prepare ImageNet-R by running:")
            print("  1. python scripts/download_datasets.py --dataset imagenet_r")
            print("  2. python scripts/split_imagenet_r.py")
        print()
        sys.exit(1)

    # 3. Hyperparameter selection
    null_eta = "0.90" if (args.dataset == "imagenet_r" and args.tasks == 10) else "0.95"
    mamba_cl_dir = project_root / "baselines" / "mamba-cl"
    train_eval_script = mamba_cl_dir / "train_eval.py"

    cmd = [
        sys.executable,
        str(train_eval_script),
        "-d", args.dataset,
        "-t", str(args.tasks),
        "-e", str(args.epochs),
        "-b", str(args.batch_size),
        "--pretrained_path", str(pretrained_path),
        "--data_root", str(data_root),
        "--null_eta", null_eta,
        "--use_null_space",
        "--seed", str(args.seed),
        "--use_amp", args.use_amp,
        "-jt", str(args.workers),
    ]

    if args.use_wandb:
        cmd.append("--use_wandb")

    if unknown_args:
        cmd.extend(unknown_args)

    print("=" * 70)
    print(" HippoCortex Continual Learning: Mamba-CL Baseline Runner")
    print("=" * 70)
    print(f" Dataset         : {args.dataset}")
    print(f" Tasks           : {args.tasks}")
    print(f" Epochs per task : {args.epochs}")
    print(f" Batch Size      : {args.batch_size}")
    print(f" Data Directory  : {data_root}")
    print(f" Model Weights   : {pretrained_path}")
    print(f" AMP (FP16)      : {args.use_amp}")
    print(f" Workers         : {args.workers}")
    print("=" * 70)
    print("Command to execute:")
    print(" ".join(cmd))
    print("=" * 70 + "\n")

    env = os.environ.copy()
    pythonpath = str(mamba_cl_dir)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = pythonpath

    ret = subprocess.run(cmd, cwd=str(mamba_cl_dir), env=env)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
