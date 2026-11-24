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

1. **Awake phase** — supervised update on the current task's data (Phase 1 warmup, then Phase 2 joint,
   where backbone gradients are projected through `P = I − UUᵀ`).
2. **Sleep phase (consolidation, Phase 3)** — `hippocortex.cl.consolidation.consolidate`. The backbone is not updated.
   a. Collect the real hidden states `H_N` of task `N` (eval mode, no gradients).
   b. Sample `Ĥ_past` for each prior task id from the SWR generator *as it is before this sleep*.
   c. Fine-tune the CVAE on `H_N` (conditioned on `N`) together with `Ĥ_past` (conditioned on their task ids),
      so it learns task `N` without forgetting how to generate the old tasks (deep generative replay).
3. After the sleep, fit `(µ_N, σ²_N)` from `H_N` into the stats buffer and update `U` with the new feature directions.

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
**Decision:** `model.encode_image(x, pre_logits=True)` of Vim-small — the final-norm output at the middle class token → shape `(B, 384)`. (With `arch: defocus_base`, the older De-focus Mamba base network, it is `(B, 768)`.)

**Why:** Vim-small is the network of both baselines (Inf-SSM, and Mamba-CL under the Inf-SSM protocol; `baselines/mamba-cl/utils/vim_mamba.py`), and this is the feature their heads read. A different network or layer makes the Table 1 comparison invalid.

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

Accuracy is **class-incremental** by default (`eval.mode: class_il`), matching Mamba-CL's `evaluate_tasks_sofar`: after task `i`, the heads of tasks `0..i` are joined into one head over all seen classes, task `j`'s local labels are shifted by the number of classes in earlier heads, and a prediction is correct only if the argmax over every seen class is right. `eval.mode: task_il` scores each task with its own head (the task identity is given); it is easier and is not comparable to the baselines.

### 8. Backbone and Phase 2 (joint) scope
**Decision:** `MambaBackbone` wraps Vim-small from `baselines/mamba-cl/utils/vim_mamba.py` (24 blocks, width 384) and loads the authors' task-0 checkpoint (`checkpoint_path: hf` → `ileening0022/Inf-SSM`, `cifar100_10task/task_0_checkpoint_seed_0.pth`), the same start as both baselines. It fine-tunes only `head`, `x_proj`, `out_proj.weight`, `A_log`, `A_b_log` (Mamba-CL `--training_string`, plus Vim's backward A). Images are bicubic-resized to 224 inside the backbone. The optimizer is rebuilt per task, as Mamba-CL does.

The protocol follows the shared-head baselines: Split-CIFAR100 **10×10 in natural class order** with ImageNet normalisation, **one shared 100-way head** (`head_mode: shared`, loaded from the task-0 checkpoint; CE over all classes seen so far, head weight decay 0), task 0 not retrained (`training.train_first_task: false`), Adam-style lr 2e-4 with the head at ×50, seed 0. `head_mode: per_task` and `arch: defocus_base` keep the older Mamba-CL-own-protocol setup (20×5, one head per task, 768-d).

Still different from the baselines by default: CIFAR images are upsampled from 32×32 with crop + flip only. `dataset.augment: inf_ssm` uses the baselines' own transforms (224 RandAugment, colour jitter, random erasing; ImageNet stats for train and CIFAR stats for eval, as in `baselines/inf-ssm/cl_datasets.py`). Mixup/cutmix and label smoothing are not implemented. Batch 64 (baselines 128), and 5 + 10 epochs per task (baselines 40).

**Known limitations of the Phase 2 loop** (open design questions — see the `trainer.py` module docstring):
- `NullSpaceProjector` holds a single basis of the feature width (384), so only `out_proj.weight` (first axis) can be projected. `x_proj_*`, `A_log` and `A_b_log` have no such axis; they are skipped (logged once, counted in the metrics) and train **unprotected**. Mamba-CL builds a separate basis per parameter from that layer's input activations.
- The head is **not** projected. With per-task heads that protects nothing; with the shared head the old classes' columns are trained by the current task's CE, which is where replay of old hidden states through the head is meant to help.
- The head learning rate is `lr × head_lr_scale` = 2e-4 × 50 = 1e-2, as Mamba-CL. The backbone lr is 2e-4.
- Gradients are projected *before* AdamW, which rescales elements and can leave the null space; Mamba-CL projects the *update*.
- Replay samples are generated in the backbone's output space, so a loss on them cannot change the backbone. The default replay term is a no-grad ELBO value kept for monitoring; `Trainer._replay_loss` is the one place to change it.

---

## Baseline Results

Source of truth: W&B entity `hasinthakapiyumal-uok` (Kaggle notebooks are saved without outputs).
Regenerate the CSV (`results/stage1_split_cifar100/baselines.csv`, gitignored) with:

```bash
python experiments/stage1_split_cifar100/eval_baseline.py   --wandb_run hasinthakapiyumal-uok/seq-finetune-baseline/aab59762ef36 --method seq   --wandb_run hasinthakapiyumal-uok/inf-ssm-baseline/d591d88c21e6 --method inf_ssm   --wandb_run hasinthakapiyumal-uok/mamba-cl-baseline/3388e092e0b1 --method mamba_cl_shared_head   --wandb_run hasinthakapiyumal-uok/mamba-cl-baseline/3427efba80b5 --method mamba_cl
```

### Main comparison: Inf-SSM protocol (Split-CIFAR100, 10 tasks × 10 classes)

Vim-small, one shared 100-class head, authors' task-0 checkpoint, 40 epochs/task, batch 128, seed 0.
All three run from `baselines/inf-ssm` (`--cl_method seq | reg | mamba_cl`).

| Method | AA ↑ | AIA ↑ | FM ↓ | TIL ↑ | W&B run | Date |
|--------|------|-------|------|-------|---------|------|
| Sequential fine-tune (lower bound) | 21.56 | 48.89 | 39.28 | 62.76 | `seq-finetune-baseline/aab59762ef36` | 2026-11-22 |
| Inf-SSM (λ = 3×10⁴) | 28.42 | 55.35 | 24.83 | 70.99 | `inf-ssm-baseline/d591d88c21e6` | 2026-10-22 |
| Mamba-CL (η = 0.95, `--optim mamba_cl`, head wd 0) | 51.52 | 59.43 | 48.62 | 94.94 | `mamba-cl-baseline/3388e092e0b1` | 2026-11-10 |
| **HippoCortex** | — | — | — | — | — | — |

- Inf-SSM paper (Table 2, mean of seeds 0/10/100): AA 26.53 ± 3.10, AIA 54.24 ± 1.87, FM 24.00 ± 1.80. Ours is within one std.
- Inf-SSM paper Seq row: AA 20.58, AIA 51.37, FM 71.49. Our AA/AIA are close; our FM is much lower — check the FM definition before quoting it.
- Mamba-CL forgets in the shared head, not the features: old tasks keep ~95% TIL.
- Still to run: seeds 10 and 100 for all three; ImageNet-R.

### Mamba-CL, its own protocol (not comparable with the table above)

Separate 10-class head per task (joined at test), `baselines/mamba-cl`, Vim-small, task-0 start `inf_ssm_task0`, seed 0:
AA 79.52, FM 8.53 (`mamba-cl-baseline/3427efba80b5`, 2026-11-01).
Older runs on other backbones: Defocus-Mamba large (21K) 10×10 AA 90.16 / FM 2.18 (2026-10).

--------|----------|-------|----|----|--------|
| Mamba-CL | large (21K) | 10×10 | **90.16%** | **2.18%** | reproduced run (Hasinthaka, 2026-10) |
| Mamba-CL | base (1K)   | 20×5  | *(pending, Kaggle run)* | *(pending)* | — |
| Inf-SSM  | base (1K)   | 20×5  | *(pending, Hasinthaka)* | *(pending)* | — |
| EWC | — | Split-CIFAR100 | — | — | — |
| DGR | — | Split-CIFAR100 | — | — | — |
| **HippoCortex** | — | Split-CIFAR100 | — | — | — |

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
