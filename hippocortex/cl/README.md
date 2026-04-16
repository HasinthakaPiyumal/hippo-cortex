# `hippocortex/cl/`

Continual-learning primitives — the machinery that makes catastrophic forgetting go away without raw-sample replay.

## Planned contents

- **Stats buffer** — per-task `(µ, σ²)` of Mamba hidden states. Append-only; constant memory footprint regardless of task count.
- **Null-space projector** — builds/maintains `U` (orthonormal basis of the feature subspace used by prior tasks) and applies `P = I − UUᵀ` to deflect new-task gradients away from previously learned representations (Saha et al. 2021).
- **Consolidation step** — wires the SWR generator (sampling `Ĥ_past` from the stats buffer) into the projected update during the "sleep" phase.

## References

- Saha, Garg, Roy. "Gradient Projection Memory for Continual Learning." ICLR 2021.
- Cheng et al. "Mamba-CL: Optimizing Selective State Space Model in Null Space for Continual Learning." arXiv:2411.15469, 2024.
- Lee et al. "Exemplar-Free Continual Learning for State Space Models." arXiv:2505.18604, 2025.
