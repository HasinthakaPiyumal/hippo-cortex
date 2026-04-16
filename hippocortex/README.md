# `hippocortex/` — main Python package

Everything that ends up as importable Python lives here. One unified package spans both stages so Stage-1 building blocks (stats buffer, cVAE SWR generator, null-space projector, Mamba backbone) are reused unchanged when Stage 2 wraps the hybrid Mamba+Transformer stack.

## Module map

| Module        | Purpose                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------- |
| `models/`     | Mamba SSM backbone, conditional-VAE SWR generator, hybrid Mamba+Transformer vision stack.    |
| `cl/`         | Continual-learning machinery: statistics buffer `(µ, σ²)`, null-space projector `I − UUᵀ`.   |
| `data/`       | Dataset/dataloader code: Split-CIFAR100, ImageNet-R, and robot sensor-stream adapters.       |
| `training/`   | Train loops, losses, optimisers, schedulers, consolidation phase driver.                     |
| `robot/`      | Go2 Edu integration: Jetson Orin NX runtime, RealSense D435i, L1 LiDAR, IMU, ROS2 nodes.     |
| `utils/`      | Logging, metrics (forgetting / backward transfer), seeding, device placement, config utils.  |

## Conventions

- Python 3.10+ (decision pending — revisit when `pyproject.toml` lands).
- PyTorch as the tensor/autograd backend; ROS2 only in `robot/`.
- Public API lives in each submodule's `__init__.py` once we add one.
- No circular imports between `models/`, `cl/`, and `training/` — `training/` may import from the other two, not the reverse.
