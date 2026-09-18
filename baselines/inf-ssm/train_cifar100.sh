#!/usr/bin/env bash
# Default to /tmp/cifar100-images if it exists, otherwise fallback to project data directory
if [ -z "${DATA_ROOT}" ]; then
    if [ -d "/tmp/cifar100-images" ]; then
        DATA_ROOT="/tmp/cifar100-images"
    else
        DATA_ROOT="../../data/cifar100-images"
    fi
fi

# Default pretrained weight path
if [ -z "${PRETRAINED_PATH}" ]; then
    if [ -f "./defocus_mamba_large_cls_21k.pth" ]; then
        PRETRAINED_PATH="./defocus_mamba_large_cls_21k.pth"
    elif [ -f "../../defocus_mamba_large_cls_21k.pth" ]; then
        PRETRAINED_PATH="../../defocus_mamba_large_cls_21k.pth"
    else
        PRETRAINED_PATH="../../defocus_mamba_large_cls_21k.pth"
    fi
fi

python -u train_eval.py \
    -d cifar100 \
    -t 10 \
    --pretrained_path "${PRETRAINED_PATH}" \
    --data_root "${DATA_ROOT}" \
    --use_inf_ssm True \
    --inf_ssm_lambda 2e5 \
    --seed 2024 \
    -b 800 \
    --use_amp True \
    -jt 4 \
    --use_wandb \
    "$@"