# HippoCortex architecture

Living design doc. Update this when the architecture changes — the proposal is a snapshot in time, this file is the moving truth.

## Biological mapping (proposal Figure 1)

```
              Awake Phase                    Sleep / SWR Phase
  ┌──────────────┐     ┌─────────────┐   SWR burst   ┌──────────────┐
  │    New       │ ──▶ │ Hippocampus │ ────────────▶ │  Neocortex   │
  │  Experience  │     │ fast encoder│               │ long-term    │
  └──────────────┘     └─────────────┘               │   store      │
                                                     └──────────────┘
```

| Biology                   | HippoCortex analogue                                            |
| ------------------------- | --------------------------------------------------------------- |
| Hippocampus fast encoder  | Stats buffer `(µ, σ²)` of Mamba hidden states (per task).       |
| SWR burst                 | Conditional-VAE SWR generator sampling `Ĥ_past`.                |
| Neocortex long-term store | Mamba backbone weights updated via null-space-projected grads.  |

## Stage 1: core algorithm (proposal Figure 2)

```
                    ┌───────────┐   Ĥ_past   ┌──────────────┐
  Task N ─────────▶ │ Stats     │ ─────────▶ │ SWR          │
                    │ Buffer    │            │ Generator    │
                    │ (µ, σ²)   │            │ (cond. VAE)  │
                    └───────────┘            └──────┬───────┘
                                                    │
                                                    ▼
                    ┌─────────────┐    ∇⊥   ┌───────────────┐
  ──────────────▶   │ Mamba       │ ◀─────  │ Null-Space    │ ──▶ Output
                    │ Network     │         │ Projector     │
                    │             │         │ P = I − UUᵀ   │
                    └─────────────┘         └───────────────┘
```

### Components

- **Stats buffer** — stores per-task `(µ, σ²)` of Mamba hidden states. No raw samples. Memory footprint is constant regardless of task count.
- **SWR generator** — conditional VAE (Sohn et al. 2015) conditioned on task id, trained to sample `Ĥ_past` from `(µ, σ²)`.
- **Null-space projector** — `P = I − UUᵀ` where `U` spans the feature subspace of prior tasks (Saha et al. 2021). Deflects new-task gradients away from directions already used.
- **Mamba backbone** — selective SSM (Gu & Dao 2023) that produces hidden states consumed by all three above.

### Training loop

1. **Awake phase** — supervised update on the current task's data.
2. **Sleep phase (consolidation)**
   a. Sample `Ĥ_past` from the SWR generator conditioned on each prior task id.
   b. Compute loss on the synthetic past.
   c. Project the gradient through `P = I − UUᵀ` before applying.
3. After task `N` completes, fit new `(µ_N, σ²_N)` and update `U` with the new feature directions.

## Stage 2: hybrid vision architecture (proposal Figure 3)

```
  Input    ┌───────┐   ┌─────────────┐   ┌───────┐
  Stream ─▶│ Mamba │ ─▶│ Transformer │ ─▶│ Mamba │ ─▶ Output
           │       │   │    Attn     │   │       │
           └───────┘   └─────────────┘   └───────┘
           └────────── HippoCortex SWR + Null-Space Projector ──────────┘
```

- Mamba layers handle **efficient temporal state tracking** across sensor streams.
- Transformer attention extracts **rich spatial features** from RGB + D435i depth frames.
- The HippoCortex SWR + projector module wraps the full stack — continual learning applies across both layer types without architectural modification.

## Invariants

- **Zero raw data storage** at any point in the pipeline.
- **Constant memory footprint** regardless of how many tasks have been learned.
- **Linear-time inference** preserved end-to-end (Mamba property; Transformer attention is bounded by Stage-2 context length).
- **On-board inference** for Stage 2 — everything must fit on the Jetson Orin NX 16GB (100 TOPS).
