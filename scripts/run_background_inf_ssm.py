# scripts/run_background_inf_ssm.py
import subprocess
import os
import argparse
import sys
import shutil

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

parser = argparse.ArgumentParser(description="HippoCortex Background Runner for Inf-SSM Baseline")
parser.add_argument("-d", "--dataset", type=str, default="cifar100", choices=["cifar100", "imagenet_r", "sdomainet"])
parser.add_argument("-t", "--tasks", type=int, default=10, help="Number of tasks (e.g. 10 or 20)")
parser.add_argument("-b", "--batch_size", type=int, default=800, help="Batch size (800 for CIFAR100, 256 for ImageNet-R)")
parser.add_argument("--lambda_val", type=float, default=2e5, help="Regularization strength for Inf-SSM")
parser.add_argument("--data_root", type=str, default="", help="Data root (defaults to fast local /tmp or project data)")
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--use_amp", type=str, default="True")
parser.add_argument("-jt", "--workers", type=int, default=4)
parser.add_argument("--use_wandb", action="store_true", default=True, help="Enable wandb logging")
parser.add_argument("--wandb_project", type=str, default="inf-ssm-baseline", help="WandB project name")
parser.add_argument("--wandb_name", type=str, default="", help="WandB run name")
args = parser.parse_args()

# Optional: WandB login if WANDB_API_KEY environment variable is present
if os.environ.get("WANDB_API_KEY"):
    try:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"])
    except Exception as e:
        print(f"Notice: WandB auto-login skipped ({e})")

# Determine data root default if not specified
if not args.data_root:
    if args.dataset == "cifar100":
        if os.path.exists("/tmp/cifar100-images"):
            args.data_root = "/tmp/cifar100-images"
        elif os.path.exists(os.path.join(project_root, "data/cifar100-images")):
            # Try to copy to /tmp for faster I/O if /tmp exists
            if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
                print("Copying data/cifar100-images to /tmp/cifar100-images for fast caching...")
                try:
                    shutil.copytree(
                        os.path.join(project_root, "data/cifar100-images"),
                        "/tmp/cifar100-images",
                        dirs_exist_ok=True
                    )
                    args.data_root = "/tmp/cifar100-images"
                except Exception:
                    args.data_root = os.path.join(project_root, "data/cifar100-images")
            else:
                args.data_root = os.path.join(project_root, "data/cifar100-images")
        else:
            args.data_root = os.path.join(project_root, "data/cifar100-images")
    elif args.dataset == "imagenet_r":
        if os.path.exists("/tmp/imagenet-r"):
            args.data_root = "/tmp/imagenet-r"
        else:
            args.data_root = os.path.join(project_root, "data/imagenet-r")
    else:
        args.data_root = os.path.join(project_root, f"data/{args.dataset}")

# Determine python executable
venv_python = os.path.join(project_root, ".venv/bin/python")
python_bin = venv_python if os.path.exists(venv_python) else sys.executable

pretrained_path = os.path.join(project_root, "defocus_mamba_large_cls_21k.pth")
if not os.path.exists(pretrained_path):
    print(f"Warning: Pretrained checkpoint not found at {pretrained_path}")

cmd = [
    python_bin,
    "-u",  # Unbuffered output for real-time log flushing with nohup / background tasks
    os.path.join(project_root, "baselines/inf-ssm/train_eval.py"),
    "-d", args.dataset,
    "-t", str(args.tasks),
    "--pretrained_path", pretrained_path,
    "--data_root", args.data_root,
    "--use_inf_ssm", "True",
    "--inf_ssm_lambda", str(args.lambda_val),
    "--seed", str(args.seed),
    "-b", str(args.batch_size),
    "--use_amp", args.use_amp,
    "-jt", str(args.workers),
]

if args.use_wandb:
    cmd.append("--use_wandb")
    cmd.extend(["--wandb_project", args.wandb_project])
    if args.wandb_name:
        cmd.extend(["--wandb_name", args.wandb_name])

log_path = os.path.join(project_root, f"training_inf_ssm_{args.dataset}.log")

with open(log_path, "w") as log_file:
    p = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=project_root
    )
    print(f"Background Inf-SSM training started successfully ({args.dataset}, {args.tasks} tasks)!")
    print(f"Process ID (PID): {p.pid}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Outputs are being written to: {log_path}")
    print(f"To monitor output live, run:\n  tail -f {log_path}")
