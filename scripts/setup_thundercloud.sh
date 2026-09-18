#!/usr/bin/env bash
# scripts/setup_thundercloud.sh
# Complete environment setup script for Thundercloud remote GPU instances.
# Usage:
#   bash scripts/setup_thundercloud.sh [OPTIONAL_WANDB_API_KEY]

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "======================================================================="
echo " Starting Hippo-Cortex Environment Setup for Thundercloud"
echo " Repo root: ${REPO_ROOT}"
echo "======================================================================="

# 1. System packages & Python 3.11
echo "[1/7] Updating apt repositories and installing Python 3.11..."
sudo apt update
sudo apt install -y software-properties-common build-essential wget git

# Add deadsnakes PPA if python3.11 is not available
if ! command -v python3.11 &> /dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
fi

sudo apt install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils python3-pip

# 2. Virtual Environment
echo "[2/7] Setting up Python 3.11 virtual environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip

# 3. PyTorch with CUDA 12.1
echo "[3/7] Installing PyTorch 2.3.1 with CUDA 12.1..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 4. Build tools and Mamba kernels
echo "[4/7] Installing build dependencies and Mamba custom CUDA kernels..."
pip install ninja packaging wheel
pip install --no-build-isolation causal-conv1d==1.4.0
pip install --no-build-isolation mamba-ssm==1.2.2

# 5. Project dependencies & package installation
echo "[5/7] Installing repository requirements..."
pip install -r requirements.txt
pip install easydict yacs
pip install -e .

# 6. Pretrained checkpoint
echo "[6/7] Checking pretrained Defocus Mamba checkpoint..."
PRETRAINED_FILE="${REPO_ROOT}/defocus_mamba_large_cls_21k.pth"
if [ ! -f "${PRETRAINED_FILE}" ]; then
    echo "Downloading defocus_mamba_large_cls_21k.pth (approx 1.2 GB)..."
    wget -O "${PRETRAINED_FILE}" https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/defocus_mamba_large_cls_21k.pth
else
    echo "Pretrained weights already present at ${PRETRAINED_FILE}."
fi

# 7. CIFAR-100 Dataset download, split, and fast /tmp caching
echo "[7/7] Preparing CIFAR-100 dataset..."
if [ ! -d "${REPO_ROOT}/data/cifar100-images" ]; then
    echo "Downloading CIFAR-100..."
    python scripts/download_datasets.py --dataset cifar100
    echo "Splitting CIFAR-100 into class directory structure..."
    python scripts/split_cifar100.py
fi

if [ ! -d "/tmp/cifar100-images" ]; then
    echo "Copying CIFAR-100 dataset to /tmp/cifar100-images for fast local I/O..."
    cp -r "${REPO_ROOT}/data/cifar100-images" /tmp/
fi

# Optional: WandB authentication
WANDB_KEY="${1:-$WANDB_API_KEY}"
if [ -n "${WANDB_KEY}" ]; then
    echo "Logging in to Weights & Biases..."
    export WANDB_API_KEY="${WANDB_KEY}"
    wandb login "${WANDB_KEY}"
fi

echo "======================================================================="
echo " Setup complete! Virtual environment ready at ${REPO_ROOT}/.venv"
echo " To run Inf-SSM training:"
echo "   bash scripts/run_inf_ssm_cifar100.sh"
echo "======================================================================="
