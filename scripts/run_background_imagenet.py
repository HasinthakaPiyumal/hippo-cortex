# scripts/run_background_imagenet.py
import subprocess
import os
import argparse

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

parser = argparse.ArgumentParser(description="HippoCortex Background Runner for ImageNet-R")
parser.add_argument("-t", "--tasks", type=int, default=10, choices=[10, 20], help="Number of tasks (10 or 20)")
args = parser.parse_args()

# Select null_eta based on tasks count (matching train_imagenet_r.sh and train_imagenet_r-s20.sh)
null_eta = "0.90" if args.tasks == 10 else "0.95"

cmd = [
    os.path.join(project_root, ".venv/bin/python"),
    os.path.join(project_root, "baselines/mamba-cl/train_eval.py"),
    "-d", "imagenet_r",
    "-t", str(args.tasks),
    "--pretrained_path", os.path.join(project_root, "defocus_mamba_large_cls_21k.pth"),
    "--data_root", "/tmp/imagenet-r",  # Use fast local SSD
    "--null_eta", null_eta,
    "--use_null_space",
    "--seed", "2024",
    "-b", "256",                         # Safe batch size for 192x192 resolution on A6000
    "--use_amp", "True",                  # Enable mixed precision (float16)
    "-jt", "4",                           # Optimize CPU loader workers
    "--use_wandb"
]

log_path = os.path.join(project_root, "training_imagenet.log")

with open(log_path, "w") as log_file:
    p = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=project_root
    )
    print(f"Background ImageNet-R training started successfully ({args.tasks} tasks)!")
    print(f"Process ID (PID): {p.pid}")
    print(f"Outputs are being written to: {log_path}")
