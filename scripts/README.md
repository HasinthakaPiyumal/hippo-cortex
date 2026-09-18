# `scripts/`

Utility, setup, and orchestration scripts for the HippoCortex continual learning framework and baselines.

---

## Running on Modal.com (Serverless Cloud GPU)

[Modal](https://modal.com) allows you to launch training on cloud GPUs (A100, A10G, L4) with zero server maintenance.

### 1. Install Modal & Authenticate
On your local terminal:
```bash
pip install modal
modal setup
```
*(This opens a browser tab to connect your free or paid Modal account).*

### 2. Set Up WandB (Optional)
If you want live WandB logging:
```bash
# Set in your local environment:
export WANDB_API_KEY="your_wandb_api_key"
```

### 3. Launch Inf-SSM on Modal Cloud GPU
Run Split-CIFAR100 on an NVIDIA A100-40GB GPU:
```bash
modal run scripts/run_modal_inf_ssm.py --dataset cifar100 --batch-size 800 --gpu A100-40GB
```

Or on an NVIDIA A10G (cost-efficient, ~$1.10/hr):
```bash
modal run scripts/run_modal_inf_ssm.py --dataset cifar100 --batch-size 800 --gpu A10G
```

### How it works under the hood:
1. **Zero-build Docker container**: Uses verified pre-built Linux wheels for `causal-conv1d` and `mamba-ssm`, starting up in ~30 seconds.
2. **Persistent cloud caching**: Model weights (`defocus_mamba_large_cls_21k.pth`) and datasets (`cifar100`) are automatically downloaded once and saved to a persistent `modal.Volume("hippo-cortex-data")`.
3. **High-speed NVMe scratch**: Data is automatically staged into `/tmp/` on the GPU machine for maximum dataloading speed during training.
4. **VRAM optimized**: Uses the chunked `bar_A` state extraction and AMP mixed precision so batch size 800 runs cleanly.

---

## Local Background Runner

- `scripts/run_background_inf_ssm.py`: Detached background runner for local Linux machines using `nohup`.

---

## Dataset Preparation

- `scripts/download_datasets.py`: Download raw datasets:
  ```bash
  python scripts/download_datasets.py --dataset cifar100
  python scripts/download_datasets.py --dataset imagenet_r
  ```
- `scripts/split_cifar100.py`: Split raw CIFAR-100 python batches into image folders under `data/cifar100-images/`.
- `scripts/split_imagenet_r.py`: Split ImageNet-R directory into train/val structure.
