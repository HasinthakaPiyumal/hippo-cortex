# `scripts/`

Utility, setup, and orchestration scripts for the HippoCortex continual learning framework and baselines.

---

## Thundercloud (Remote GPU Instance) Setup & Run

### 1. Automated Setup
To configure a fresh Ubuntu GPU instance with Python 3.11, PyTorch 2.3.1 (cu121), Mamba custom CUDA kernels (`causal-conv1d`, `mamba-ssm`), checkpoint weights, and CIFAR-100 data:
```bash
git clone https://github.com/HasinthakaPiyumal/hippo-cortex.git
cd hippo-cortex

# Run automated setup (optionally pass your WandB API key)
bash scripts/setup_thundercloud.sh [OPTIONAL_WANDB_API_KEY]
```

### 2. Launch Inf-SSM Baseline (Split-CIFAR100)
Run in the background with `nohup`:
```bash
bash scripts/run_inf_ssm_cifar100.sh
```

To monitor live output:
```bash
tail -f training_inf_ssm_cifar100.log
```

---

## Background & Cloud Runners

- `scripts/run_inf_ssm_cifar100.sh`: Shell script launching Inf-SSM on CIFAR-100 via `nohup` with preconfigured hyperparameters (`b=800`, `amp=True`, `workers=4`, `use_wandb`).
- `scripts/run_background_inf_ssm.py`: Python detached background runner supporting custom datasets (`cifar100`, `imagenet_r`, `sdomainet`), task counts, and batch sizes.
- `scripts/run_modal_inf_ssm.py`: Serverless GPU execution on [Modal.com](https://modal.com) with cloud volume caching.

---

## Dataset Preparation

- `scripts/download_datasets.py`: Download raw datasets:
  ```bash
  python scripts/download_datasets.py --dataset cifar100
  python scripts/download_datasets.py --dataset imagenet_r
  ```
- `scripts/split_cifar100.py`: Split raw CIFAR-100 python batches into image folders under `data/cifar100-images/`.
- `scripts/split_imagenet_r.py`: Split ImageNet-R directory into train/val structure.
