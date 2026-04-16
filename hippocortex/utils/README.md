# `hippocortex/utils/`

Cross-cutting helpers — import from anywhere, import from nothing inside the package.

## Planned contents

- **Logging** — structured logger + experiment-tracker shim (wandb / tensorboard / mlflow — decision pending).
- **Metrics** — `average_accuracy`, `average_forgetting`, `backward_transfer`, plus robot-side metrics (latency, throughput, per-sensor drops).
- **Seeding** — reproducible seed setter for `torch`, `numpy`, `random`, CUDA.
- **Device** — pick CUDA / MPS / CPU; Jetson-aware helpers for Stage 2.
- **Config** — thin loader around YAML/JSON configs from `experiments/*/`.
