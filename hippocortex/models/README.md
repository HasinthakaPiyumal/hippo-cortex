# `hippocortex/models/`

Network architectures.

## Planned contents

- **Mamba backbone** — selective SSM sequence model per Gu & Dao (2023). Stage 1 uses this alone; Stage 2 keeps it for temporal state tracking on robot sensor streams.
- **SWR generator** — conditional VAE (Sohn et al. 2015) that samples synthetic past hidden states `Ĥ_past` given the per-task `(µ, σ²)` stored in `hippocortex/cl/`.
- **Hybrid vision stack** — Stage 2's Mamba → Transformer-attention → Mamba pipeline for RGB + D435i depth frames.

## Not here

- Training code — see `../training/`.
- Feature-subspace tracking `U` and the projector — see `../cl/`.
