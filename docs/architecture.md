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

---

## Implementation Decisions

Locked decisions — do not change without updating this section and notifying the full team.

### 1. Hidden State Extraction Point
**Decision:** Output of the last SSM layer (index `n_layers - 1`), mean-pooled across sequence length → shape `(B, d_model)`.

**Why:** Matches Mamba-CL's extraction point exactly. Using a different layer makes Table 1 comparison invalid (Cheng et al. 2025, Section 3.2).

### 2. Null-Space Projector Rank Budget
**Decision:** `rank_budget = 200` singular vectors max (GPM default). Same value used in Mamba-CL.

**Why:** Prevents null space exhaustion at task 15–20 of Split-CIFAR100. If `d_model=128`, k=200 is over-complete and will truncate — this is expected behaviour.

### 3. CVAE Architecture (SWR Generator)
**Decision:**
- Encoder: `MLP(d_model + 32 → 512 → 512 → [µ, log_var])`, µ/log_var dim = 64
- Decoder: `MLP(64 + 32 → 512 → 512 → d_model)`
- Task embedding: `nn.Embedding(n_tasks, 32)`, concatenated to input
- Loss: ELBO = MSE recon + KL divergence (β = 1.0)

**Why:** Derived from Sohn et al. 2015. Input is hidden states (not pixels) so MLP decoder is correct.

### 4. Mamba Version
**Decision:** `mamba-ssm==1.2.2` (Mamba 1) for Stage 1. Do NOT install `mamba-ssm>=2.0.0`.

**Why:** The null-space projector (Mamba-CL lineage) targets `x_proj`, `A_log`, `out_proj.weight` — parameter names that only exist in Mamba 1. Mamba 2 (SSD architecture) eliminates `x_proj` entirely (merged into the SSD layer) and changes `A_log` structure. Installing Mamba 2 also breaks the `selective_scan_interface` API that Mamba-CL's `mamba_block.py` imports.

Mamba 2 is faster (3–8×) and planned for Stage 2 (hybrid architecture on Jetson), but switching now with a 3.5-week Paper 1 deadline is unnecessary risk.

**Verify correct version:** `python -c "import mamba_ssm; print(mamba_ssm.__version__)"` → must print `1.2.2`.

### 5. Config Format
**Decision:** YAML via Hydra + OmegaConf. Zero hardcoded hyperparameters in model or training files.

### 5. Experiment Tracking
**Decision:** Weights & Biases, project `hippocortex-stage1`. Run name format: `{method}_{dataset}_{seed}` e.g. `hippocortex_splitcifar100_42`.

### 6. ImageNet-R Task Split
**Decision:** Must use Mamba-CL's class-to-task assignment file verbatim. Induwara pulls this before writing `imagenet_r.py`.

### 7. `acc_matrix` Convention
`acc_matrix[i, j]` = accuracy on task `j` evaluated after training on task `i`. Shape `(n_tasks, n_tasks)`. Upper triangle is undefined (task `j` not yet seen).

---

## Baseline Results

*(Filled by Thagya after running `experiments/stage1_split_cifar100/eval_baseline.py`)*

| Method | Dataset | AA | AF | BWT | Memory (MB) | Commit / Date |
|--------|---------|----|----|-----|-------------|---------------|
| Mamba-CL | Split-CIFAR100 | — | — | — | — | — |
| Inf-SSM | Split-CIFAR100 | — | — | — | — | — |
| EWC | Split-CIFAR100 | — | — | — | — | — |
| DGR | Split-CIFAR100 | — | — | — | — | — |
| **HippoCortex** | Split-CIFAR100 | — | — | — | — | — |

---

## Module Dependency Graph

```
experiments/run.py
    └── hippocortex.training.trainer.Trainer
            ├── hippocortex.models.backbone.MambaBackbone        (Praveen)
            ├── hippocortex.models.swr_generator.SWRGenerator    (Praveen)
            ├── hippocortex.cl.stats_buffer.StatsBuffer          (Hasinthaka)
            ├── hippocortex.cl.null_space_projector.NSP          (Praveen)
            └── hippocortex.cl.consolidation.consolidate         (Hasinthaka)

hippocortex.data.split_cifar100  — zero internal deps (Induwara)
hippocortex.data.imagenet_r      — zero internal deps (Induwara)
hippocortex.utils.metrics        — zero internal deps (Thagya)
hippocortex.utils.memory_tracker — zero internal deps (Thagya)
```
