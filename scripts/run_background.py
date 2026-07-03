# scripts/run_background.py
import subprocess
import os

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

cmd = [
    os.path.join(project_root, ".venv/bin/python"),
    os.path.join(project_root, "baselines/mamba-cl/train_eval.py"),
    "-d", "cifar100",
    "--pretrained_path", os.path.join(project_root, "defocus_mamba_large_cls_21k.pth"),
    "--data_root", "/tmp/cifar100-images",  # Use fast local SSD
    "--null_eta", "0.95",
    "--use_null_space",
    "--seed", "2024",
    "--use_amp", "True",                     # Enable mixed precision (float16)
    "-jt", "4",                              # Optimize CPU loader workers
    "--use_wandb"
]

log_path = os.path.join(project_root, "training.log")

with open(log_path, "w") as log_file:
    # Start process detached
    p = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=project_root
    )
    print(f"Background training started successfully with optimized settings!")
    print(f"Process ID (PID): {p.pid}")
    print(f"Outputs are being written to: {log_path}")
