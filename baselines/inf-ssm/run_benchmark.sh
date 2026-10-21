#!/usr/bin/env bash
# Inf-SSM on Vim-small with the paper hyperparameters (official vim/scripts/*.sh), one run per seed.
#   bash run_benchmark.sh cifar100 10            # seeds 0 10 100
#   bash run_benchmark.sh imagenet_r 5 0         # seed 0 only
# DATA_ROOT: CIFAR-100 download dir, or the extracted imagenet-r / 256_ObjectCategories folder.
# EXTRA: more train_eval.py flags, e.g. EXTRA="--use_wandb --grad_ckpt true".
# Seed 0 starts from the authors' task-0 checkpoint; the authors released no other seeds, so those
# train task 0 themselves from the ImageNet Vim-small weights (--full_run).
set -euo pipefail
DATASET=${1:?dataset: cifar100 | imagenet_r | caltech256}
TASKS=${2:?num tasks: 5 | 10}
shift 2
SEEDS=("${@:-0 10 100}")
SEEDS=(${SEEDS[@]})
DATA_ROOT=${DATA_ROOT:-../../data/${DATASET}}
CKPT_DIR=${CKPT_DIR:-./checkpoints}
LOG_DIR=${LOG_DIR:-./logs}
mkdir -p "${LOG_DIR}"

for SEED in "${SEEDS[@]}"; do
    if [ "${SEED}" = "0" ]; then INIT=(--finetune hf); else INIT=(--finetune vim_small_in1k --full_run); fi
    LOG="${LOG_DIR}/inf_ssm_${DATASET}_t${TASKS}_seed_${SEED}.log"
    echo "seed ${SEED} -> ${LOG}"
    python -u train_eval.py -d "${DATASET}" -dr "${DATA_ROOT}" -t "${TASKS}" --seed "${SEED}" "${INIT[@]}" \
        --cl_method reg --reg_mode inf_ssm --ckpt_dir "${CKPT_DIR}" ${EXTRA:-} > "${LOG}" 2>&1
done
