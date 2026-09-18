#!/usr/bin/env bash
# scripts/run_inf_ssm_cifar100.sh
# One-command background runner for Inf-SSM on Split-CIFAR100 using nohup.
# Usage:
#   bash scripts/run_inf_ssm_cifar100.sh [OPTIONAL_ARGS]

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Optional WandB login
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
fi

# Ensure /tmp/cifar100-images exists for high-speed I/O
DATA_ROOT="/tmp/cifar100-images"
if [ ! -d "${DATA_ROOT}" ]; then
    if [ -d "data/cifar100-images" ]; then
        echo "Copying data/cifar100-images to /tmp/cifar100-images for fast caching..."
        cp -r data/cifar100-images /tmp/
    else
        echo "Warning: /tmp/cifar100-images not found. Checking data/cifar100-images..."
        DATA_ROOT="data/cifar100-images"
    fi
fi

# Ensure pretrained checkpoint exists
PRETRAINED="./defocus_mamba_large_cls_21k.pth"
if [ ! -f "${PRETRAINED}" ]; then
    echo "Downloading defocus_mamba_large_cls_21k.pth..."
    wget -O "${PRETRAINED}" https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/defocus_mamba_large_cls_21k.pth
fi

LOG_FILE="training_inf_ssm_cifar100.log"

echo "======================================================================="
echo " Launching Inf-SSM CIFAR-100 Training in Background (nohup)"
echo " Data root:       ${DATA_ROOT}"
echo " Pretrained path: ${PRETRAINED}"
echo " Log file:        ${REPO_ROOT}/${LOG_FILE}"
echo "======================================================================="

nohup python -u baselines/inf-ssm/train_eval.py \
    -d cifar100 \
    -t 10 \
    --pretrained_path "${PRETRAINED}" \
    --data_root "${DATA_ROOT}" \
    --use_inf_ssm True \
    --inf_ssm_lambda 2e5 \
    --seed 2024 \
    -b 800 \
    --use_amp True \
    -jt 4 \
    --use_wandb \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

PID=$!

echo "Successfully started Inf-SSM training!"
echo "PID: ${PID}"
echo ""
echo "To monitor live training output, run:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check process status:"
echo "  ps aux | grep train_eval.py"
echo "======================================================================="
