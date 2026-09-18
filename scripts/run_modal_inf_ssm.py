"""
Modal.com Serverless GPU Runner for HippoCortex Inf-SSM Baseline

Prerequisites:
    1. Install Modal:
       pip install modal
    2. Authenticate your Modal account:
       modal setup
    3. Run on cloud GPU:
       modal run scripts/run_modal_inf_ssm.py --dataset cifar100 --batch-size 800 --gpu A100-40GB

Supported GPUs:
    --gpu A100-40GB   (NVIDIA A100 40GB, ~$3.67/hr, Recommended for batch size 800)
    --gpu A10G        (NVIDIA A10G 24GB, ~$1.10/hr, Cost-efficient)
    --gpu L4          (NVIDIA L4 24GB, ~$0.80/hr, Budget-friendly)
"""
import os
import subprocess
import modal

# ---------------------------------------------------------------------------
# 1. Container Image Definition with Prebuilt Mamba & Causal-Conv1D CUDA Kernels
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    # PyTorch 2.3.1 with CUDA 12.1
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Verified pre-built Linux wheels for CUDA 12 + PyTorch 2.3 on Python 3.11
    .pip_install(
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    )
    # Pinned transformers (4.41.2 strictly required for Mamba-CL) and dependencies
    .pip_install(
        "transformers==4.41.2",
        "timm==1.0.9",
        "einops==0.8.0",
        "scipy==1.13.1",
        "scikit-learn==1.5.0",
        "wandb==0.17.3",
        "easydict",
        "yacs",
        "tqdm",
        "gdown",
    )
    # Sync repository code into container
    .add_local_dir(
        ".",
        remote_path="/root/hippo-cortex",
        ignore=[".venv", ".git", "wandb", "__pycache__", "*.pth", "*.tar*", "data/*", "results/*"],
    )
    .env({
        "PYTHONPATH": "/root/hippo-cortex:/root/hippo-cortex/baselines/inf-ssm",
        "PYTHONUNBUFFERED": "1",
    })
)

# ---------------------------------------------------------------------------
# 2. Modal App and Persistent Storage Volume
# ---------------------------------------------------------------------------
app = modal.App("hippocortex-inf-ssm")
volume = modal.Volume.from_name("hippo-cortex-data", create_if_missing=True)


# ---------------------------------------------------------------------------
# 3. Core Training Routine
# ---------------------------------------------------------------------------
def _execute_training(
    dataset: str,
    tasks: int,
    batch_size: int,
    inf_ssm_lambda: float,
    seed: int,
    use_wandb: bool,
    wandb_key: str,
    wandb_project: str,
    wandb_name: str,
):
    import os
    import subprocess
    import shutil

    workdir = "/root/hippo-cortex"
    os.chdir(workdir)

    # Register local package in editable mode without reinstalling dependencies
    subprocess.run(["pip", "install", "--no-deps", "-e", "/root/hippo-cortex"], check=False)

    # Configure WandB authentication if key is provided
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        try:
            import wandb
            wandb.login(key=wandb_key)
            print("WandB login successful!")
        except Exception as e:
            print(f"Notice: WandB auto-login ({e})")

    # 1. Download / cache checkpoint on persistent cloud volume
    ckpt_path = "/vol/defocus_mamba_large_cls_21k.pth"
    if not os.path.exists(ckpt_path):
        print(f"Downloading pretrained checkpoint to persistent volume ({ckpt_path})...")
        url = "https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/defocus_mamba_large_cls_21k.pth"
        subprocess.run(["wget", "-q", "--show-progress", "-O", ckpt_path, url], check=True)
        volume.commit()
    else:
        print(f"Using cached checkpoint from persistent volume: {ckpt_path}")

    # 2. Download / cache dataset on persistent cloud volume
    vol_data_dir = f"/vol/{dataset}-images"
    if not os.path.exists(vol_data_dir):
        print(f"Preparing {dataset} dataset on persistent volume ({vol_data_dir})...")
        if dataset == "cifar100":
            # Download raw cifar100
            subprocess.run(
                ["python", "scripts/download_datasets.py", "--root", "/vol", "--dataset", "cifar100"],
                check=True
            )
            # Split raw cifar100 batches into class directories
            split_cmd = f"""
import os, pickle
from PIL import Image

RAW_DIR = '/vol/cifar100/cifar-100-python'
OUT_DIR = '{vol_data_dir}'

def unpickle(f):
    with open(f, 'rb') as fp:
        return pickle.load(fp, encoding='bytes')

for split in ('train', 'test'):
    fname = 'train' if split == 'train' else 'test'
    data = unpickle(os.path.join(RAW_DIR, fname))
    images = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = data[b'fine_labels']
    filenames = [f.decode() for f in data[b'filenames']]
    out_split = 'val' if split == 'test' else 'train'
    for img_arr, label, fn in zip(images, labels, filenames):
        folder = os.path.join(OUT_DIR, out_split, str(label))
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img_arr).save(os.path.join(folder, fn.replace('.png', '') + '.png'))
print('Dataset split and prepared on volume successfully!')
"""
            subprocess.run(["python", "-c", split_cmd], check=True)
            volume.commit()
    else:
        print(f"Using cached dataset from persistent volume: {vol_data_dir}")

    # 3. Copy to local /tmp for high-speed NVMe I/O during training
    fast_scratch_data = f"/tmp/{dataset}-images"
    if not os.path.exists(fast_scratch_data):
        print(f"Copying {vol_data_dir} to local fast NVMe disk ({fast_scratch_data})...")
        shutil.copytree(vol_data_dir, fast_scratch_data, dirs_exist_ok=True)
        print("Fast I/O cache populated.")

    # 4. Launch training command
    cmd = [
        "python", "-u", "baselines/inf-ssm/train_eval.py",
        "-d", dataset,
        "-t", str(tasks),
        "--pretrained_path", ckpt_path,
        "--data_root", fast_scratch_data,
        "--use_inf_ssm", "True",
        "--inf_ssm_lambda", str(inf_ssm_lambda),
        "--seed", str(seed),
        "-b", str(batch_size),
        "--use_amp", "True",
        "-jt", "4",
    ]

    if use_wandb:
        cmd.append("--use_wandb")
        if wandb_project:
            cmd.extend(["--wandb_project", wandb_project])
        if wandb_name:
            cmd.extend(["--wandb_name", wandb_name])

    print("=======================================================================")
    print(" Executing training in Modal cloud GPU container:")
    print(" " + " ".join(cmd))
    print("=======================================================================")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 4. Modal GPU Function Handlers
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=86400,
    volumes={"/vol": volume},
)
def train_on_a100(**kwargs):
    _execute_training(**kwargs)


@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/vol": volume},
)
def train_on_a10g(**kwargs):
    _execute_training(**kwargs)


@app.function(
    image=image,
    gpu="L4",
    timeout=86400,
    volumes={"/vol": volume},
)
def train_on_l4(**kwargs):
    _execute_training(**kwargs)


# ---------------------------------------------------------------------------
# 5. Local Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    dataset: str = "cifar100",
    tasks: int = 10,
    batch_size: int = 800,
    inf_ssm_lambda: float = 2e5,
    seed: int = 2024,
    use_wandb: bool = True,
    wandb_key: str = "",
    wandb_project: str = "inf-ssm-baseline",
    wandb_name: str = "",
    gpu: str = "A100-40GB",
):
    resolved_wandb_key = wandb_key or os.environ.get("WANDB_API_KEY", "")
    target_gpu = gpu.upper().strip()

    print(f"=== HippoCortex Inf-SSM Modal Cloud Runner ===")
    print(f"Dataset:       {dataset}")
    print(f"Tasks:         {tasks}")
    print(f"Batch Size:    {batch_size}")
    print(f"GPU:           {target_gpu}")
    print(f"Seed:          {seed}")
    print(f"Inf-SSM λ:     {inf_ssm_lambda}")
    print(f"WandB Logging: {use_wandb} (Project: {wandb_project})")
    print(f"=============================================")

    kwargs = dict(
        dataset=dataset,
        tasks=tasks,
        batch_size=batch_size,
        inf_ssm_lambda=inf_ssm_lambda,
        seed=seed,
        use_wandb=use_wandb,
        wandb_key=resolved_wandb_key,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    if "A10G" in target_gpu:
        train_on_a10g.remote(**kwargs)
    elif "L4" in target_gpu:
        train_on_l4.remote(**kwargs)
    else:
        train_on_a100.remote(**kwargs)
