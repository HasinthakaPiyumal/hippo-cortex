#!/usr/bin/env bash
# Default to /tmp/imagenet-r if it exists, otherwise fallback to project data directory
if [ -z "${DATA_ROOT}" ]; then
    if [ -d "/tmp/imagenet-r" ]; then
        DATA_ROOT="/tmp/imagenet-r"
    else
        DATA_ROOT="../../data/imagenet-r"
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
    -d imagenet_r \
    -t 10 \
    --pretrained_path "${PRETRAINED_PATH}" \
    --data_root "${DATA_ROOT}" \
    --use_inf_ssm True \
    --inf_ssm_lambda 2e5 \
    --seed 2024 \
    -b 256 \
    --use_amp True \
    -jt 4 \
    --use_wandb \
    "$@"