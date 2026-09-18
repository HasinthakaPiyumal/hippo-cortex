"""
Modal.com Runner for HippoCortex Inf-SSM Baseline

Usage:
    1. Install Modal:
       pip install modal
    2. Authenticate:
       modal setup
    3. Run on cloud GPU (e.g. A100, A10G):
       modal run scripts/run_modal_inf_ssm.py --dataset cifar100 --batch-size 400 --gpu A100-40GB
"""
import os
import subprocess
import modal

# ---------------------------------------------------------------------------
# 1. Container Image Definition with Prebuilt CUDA Kernels
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    # PyTorch 2.3.1 + CUDA 12.1
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Fast pre-built wheels for Mamba CUDA extensions (builds in seconds)
    .pip_install(
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    )
    # Additional dependencies with pinned transformers
    .pip_install(
        "timm==1.0.9",
        "transformers==4.41.2",
        "einops==0.8.0",
        "scipy==1.13.1",
        "scikit-learn==1.5.0",
        "wandb==0.17.3",
        "easydict",
        "yacs",
        "tqdm",
        "gdown",
    )
    # Sync current local hippo-cortex code into /root/hippo-cortex
    .add_local_dir(
        ".",
        remote_path="/root/hippo-cortex",
        ignore=[".venv", ".git", "wandb", "__pycache__", "*.pth", "*.tar*", "data/*", "results/*"],
    )
)

# ---------------------------------------------------------------------------
# 2. Modal App and Persistent Storage Volume
# ---------------------------------------------------------------------------
app = modal.App("hippocortex-inf-ssm")
volume = modal.Volume.from_name("hippo-cortex-data", create_if_missing=True)


# ---------------------------------------------------------------------------
# 3. Cloud GPU Execution Function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-40GB",  # Supported options: A100-40GB, A100-80GB, A10G, L40S, H100
    timeout=86400,    # Max duration: 24 hours
    volumes={"/vol": volume},
    secrets=[
        modal.Secret.from_name("my-wandb-secret")
    ] if "my-wandb-secret" in os.environ.get("MODAL_SECRETS", "") else [],
)
def train_inf_ssm(
    dataset: str = "cifar100",
    tasks: int = 10,
    batch_size: int = 400,
    inf_ssm_lambda: float = 2e5,
    seed: int = 2024,
    use_wandb: bool = True,
    wandb_key: str = "",
):
    import os

    workdir = "/root/hippo-cortex"
    os.chdir(workdir)

    # Configure WandB if key provided
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key

    # 1. Download / cache checkpoint on persistent volume
    ckpt_path = "/vol/defocus_mamba_large_cls_21k.pth"
    if not os.path.exists(ckpt_path):
        print(f"Downloading checkpoint to persistent volume at {ckpt_path}...")
        url = "https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/defocus_mamba_large_cls_21k.pth"
        subprocess.run(["wget", "-O", ckpt_path, url], check=True)
        volume.commit()
    else:
        print(f"Using cached checkpoint from {ckpt_path}")

    # 2. Download / cache dataset on persistent volume
    data_root = f"/vol/{dataset}-images"
    if not os.path.exists(data_root):
        print(f"Preparing {dataset} dataset on persistent volume at {data_root}...")
        os.makedirs(f"/vol/{dataset}_raw", exist_ok=True)
        
        if dataset == "cifar100":
            # Download raw cifar100
            subprocess.run(["python", "scripts/download_datasets.py", "--root", "/vol", "--dataset", "cifar100"], check=True)
            # Split into data_root
            split_cmd = f"""
import os, pickle
from PIL import Image

RAW_DIR = '/vol/cifar100/cifar-100-python'
OUT_DIR = '{data_root}'

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
print('Dataset preparation complete!')
"""
            subprocess.run(["python", "-c", split_cmd], check=True)
            volume.commit()
    else:
        print(f"Using cached dataset at {data_root}")

    # 3. Launch training
    cmd = [
        "python", "-u", "baselines/inf-ssm/train_eval.py",
        "-d", dataset,
        "-t", str(tasks),
        "--pretrained_path", ckpt_path,
        "--data_root", data_root,
        "--use_inf_ssm", "True",
        "--inf_ssm_lambda", str(inf_ssm_lambda),
        "--seed", str(seed),
        "-b", str(batch_size),
        "--use_amp", "True",
        "-jt", "4",
    ]
    if use_wandb:
        cmd.append("--use_wandb")

    print("Executing command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# 4. Local CLI Entry Point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    dataset: str = "cifar100",
    tasks: int = 10,
    batch_size: int = 400,
    inf_ssm_lambda: float = 2e5,
    seed: int = 2024,
    use_wandb: bool = True,
    wandb_key: str = "",
):
    print(f"Launching HippoCortex Inf-SSM on Modal Cloud GPU (dataset={dataset}, batch_size={batch_size})...")
    train_inf_ssm.remote(
        dataset=dataset,
        tasks=tasks,
        batch_size=batch_size,
        inf_ssm_lambda=inf_ssm_lambda,
        seed=seed,
        use_wandb=use_wandb,
        wandb_key=wandb_key or os.environ.get("WANDB_API_KEY", ""),
    )
