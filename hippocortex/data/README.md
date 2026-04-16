# `hippocortex/data/`

Dataset + dataloader code. Raw bytes live under the top-level `data/` directory (gitignored); this module is the code that reads them.

## Planned contents

- **Split-CIFAR100 (20 tasks)** — canonical Stage-1 benchmark; 5 classes per task.
- **ImageNet-R** — second Stage-1 benchmark.
- **Robot stream loaders** — Stage 2 adapters for the Go2 Edu:
  - RGB camera frames
  - RealSense D435i depth
  - L1 LiDAR point clouds
  - IMU samples
  - Task boundaries aligned with terrain transitions (flat floor → carpet → stairs).

## Conventions

- Datasets expose a uniform `get_task_loaders(task_id, split)` API so `training/` can iterate Stage 1 and Stage 2 task streams identically.
- No raw sample buffering across tasks — the whole point of HippoCortex.
