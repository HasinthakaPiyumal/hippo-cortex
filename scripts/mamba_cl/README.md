# Mamba-CL Baseline Runner (3 Epochs)

This directory contains standalone runners and guides to reproduce the **Mamba-CL** continual learning baseline for **3 epochs** per task on Split-CIFAR100 (or ImageNet-R).

---

## Quick Start from Scratch

### 1. Clone the Repository & Setup Environment

```bash
# Clone the repository
git clone https://github.com/HasinthakaPiyumal/hippo-cortex.git
cd hippo-cortex

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install PyTorch with CUDA support and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

### 2. Prepare Pretrained Model Weights

The Mamba backbone requires the pretrained checkpoint `defocus_mamba_large_cls_21k.pth` in the project root:

```bash
# If not already present in the root directory:
wget https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/defocus_mamba_large_cls_21k.pth
```

---

### 3. Download & Prepare CIFAR-100 Dataset

Run the automated download and directory formatting scripts:

```bash
# Download raw CIFAR-100
python scripts/download_datasets.py --dataset cifar100

# Format into train/val class folders (data/cifar100-images/)
python scripts/split_cifar100.py
```

---

### 4. Run Mamba-CL for 3 Epochs

Execute the runner script from anywhere in the repository:

```bash
python scripts/mamba_cl/run_mamba_cl_3epochs.py
```

---

## Configuration & Custom Arguments

The runner defaults to **3 epochs per task**, **10 tasks**, **batch size 200**, and **Automatic Mixed Precision (AMP)** enabled. You can customize any setting via CLI:

| Argument | Default | Description |
|---|---|---|
| `-e, --epochs` | `3` | Number of epochs to train each task |
| `-t, --tasks` | `10` | Number of continual learning tasks (e.g. `2` for a quick test) |
| `-b, --batch_size` | `200` | Batch size (default: `200`, adjust if lower VRAM is needed) |
| `-d, --dataset` | `cifar100` | Dataset name (`cifar100` or `imagenet_r`) |
| `--seed` | `2024` | Random seed for task class ordering |
| `--use_amp` | `True` | Mixed precision training for speed and low VRAM usage |
| `--use_wandb` | `False` | Enable logging to Weights & Biases |

### Examples:

- **Run with default settings (10 tasks, 3 epochs, batch size 200):**
  ```bash
  python scripts/mamba_cl/run_mamba_cl_3epochs.py
  ```

- **Quick sanity test (2 tasks, 3 epochs each):**
  ```bash
  python scripts/mamba_cl/run_mamba_cl_3epochs.py -t 2
  ```

- **Custom batch size (e.g. for lower GPU VRAM):**
  ```bash
  python scripts/mamba_cl/run_mamba_cl_3epochs.py -b 128
  ```

- **Run on ImageNet-R:**
  ```bash
  python scripts/mamba_cl/run_mamba_cl_3epochs.py -d imagenet_r -b 64
  ```

---

## Directory Contents

- `run_mamba_cl_3epochs.py`: Main cross-platform Python runner with path checking and validation.
- `README.md`: Instructions and documentation.
