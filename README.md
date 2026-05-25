# HippoCortex

**Biologically-Inspired Continual Learning via Sharp Wave Ripple Generative Replay in State Space Models**

*Group 07 — B.Sc. Hons. Software Engineering, University of Kelaniya. Supervised by Dr. Nalin Warnajith.*

---

## Overview

Standard neural networks fail at continual learning: each new task overwrites the weights that mattered for older tasks, and accuracy collapses — a failure called **catastrophic forgetting** (Kirkpatrick et al., 2017). The three existing families of fixes (rehearsal, regularisation, projection) each break down for a robot that must keep learning for months on a small on-board computer.

The mammalian brain has already solved this. The hippocampus stores fresh experience while the animal is awake, then during sleep fires brief high-frequency bursts called **sharp wave ripples (SWRs)**. These bursts send *compressed* memory traces — not raw recordings — to the neocortex for slow consolidation (Buzsáki, 2015; McClelland et al., 1995).

**HippoCortex** translates that idea into a trainable system. A Mamba SSM backbone (Gu & Dao, 2023) replaces the usual replay buffer with a small conditional VAE that recreates plausible past hidden states on demand from a per-task `(µ, σ²)` summary. Every gradient is then filtered through a null-space projector `P = I − UUᵀ` before touching any weight. No raw data is ever stored.

---

## Research Questions

| | |
|---|---|
| **RQ1** | Can a per-task `(µ, σ²)` summary of internal features replace a raw-input replay buffer without sacrificing accuracy on long task streams? |
| **RQ2** | Does a CVAE conditioned on task ID generate hidden states close enough to the true past distribution that a downstream classifier transfers within two percentage points? |
| **RQ3** | Does combining null-space gradient projection with generative replay reduce forgetting more than either mechanism alone over twenty or more sequential tasks? |
| **RQ4** | Does the integrated model fit the compute and memory budget of the Go2 Edu's Jetson Orin NX at sub-50 ms per-frame inference under a real-world task curriculum? |

---

## Objectives

- **O1 — Compact statistics buffer.** A per-task memory that keeps only `(µ, σ²)` of the backbone's hidden states; size constant in the number of samples per task.
- **O2 — SWR generator.** A Conditional VAE (Sohn et al., 2015) that, given a task ID, samples past hidden states from the buffer; classifier transfer within two percentage points of real states.
- **O3 — Null-space gradient projector.** `P = I − UUᵀ` strips each new-task gradient of components along directions important to earlier tasks; old-task accuracy drops < 3 percentage points after twenty Split-CIFAR100 tasks.
- **O4 — Standard benchmark validation.** Split-CIFAR100 (20 × 5-class tasks) and ImageNet-R; target 8–10% accuracy gain over Mamba-CL and ~4% over Inf-SSM.
- **O5 — Embodied extension.** Hybrid Mamba + Transformer stack on the Unitree Go2 Edu, with on-device Jetson Orin NX inference under 50 ms per frame.
- **O6 — Publication.** Two peer-reviewed papers: Stage 1 core algorithm and Stage 2 embodied deployment.

---

## How HippoCortex Compares

| Method | Replay type | Raw data? | Memory cost | On robot? |
|---|---|---|---|---|
| EWC (Kirkpatrick et al., 2017) | none | no | low (Fisher only) | no |
| iCaRL (Rebuffi et al., 2017) | exemplar | **yes** | grows w/ classes | no |
| DGR (Shin et al., 2017) | raw-pixel generative | no | generator weights | no |
| GPM (Saha et al., 2021) | null-space projection | no | basis matrices | no |
| Mamba-CL (Cheng et al., 2024) | null-space (SSM) | no | basis matrices | no |
| Inf-SSM (Lee et al., 2025) | extended observability | no | subspace summaries | no |
| **HippoCortex (ours)** | **SWR latent + null-space** | **no** | **compact summary per task** | **yes (Go2)** |

---

## Methodology

### Stage 1 — Core algorithm (Months 1–3)

Four components chained together:

```
Stats buffer (µ, σ²) ──▶ SWR generator (CVAE) ──▶ synthetic ĥ_past
                                                          │
Task N ──▶ Mamba backbone ──▶ loss & update ◀── null-space projector P = I − UUᵀ
```

**Training loop per new task:**
1. **Warmup** — supervised CE on the current task only; no replay, no projection.
2. **Joint** — CE on current task mixed with synthetic `ĥ_past` from the SWR generator; all gradients projected through `P` before touching any weight.
3. **Consolidation** — update the stats buffer with the new task's `(µ, σ²)` and extend `U` with new important feature directions.

**Benchmarks:** Split-CIFAR100 (20 tasks × 5 classes), ImageNet-R  
**Baselines:** Naive fine-tuning, EWC, DGR, Mamba-CL, Inf-SSM  
**Metrics:** Average accuracy, forgetting rate, memory footprint (MB) vs. task index

### Stage 2 — Embodied deployment (Months 4–6)

```
Sensor stream ──▶ Mamba ──▶ Transformer attention ──▶ Mamba ──▶ Output
└─────────────── HippoCortex SWR + Null-Space Projector ──────────────┘
```

Inference targets < 50 ms per frame on the Jetson Orin NX 16 GB (100 TOPS).  
**Four-task robot curriculum:** terrain recognition → object interaction → human-proximity awareness → new-environment mapping.

---

## Deliverables

- Open-source PyTorch implementation with reproducible training scripts and pre-trained checkpoints
- Stage 1 benchmark report: average accuracy, forgetting rate, memory footprint, full ablations
- **Stage 1 paper** (peer-reviewed) — target mid-June 2026
- Hybrid Mamba + Transformer extension for richer visual input
- Embodied demonstration on the Go2 Edu (four-task curriculum, on-device)
- **Stage 2 paper and thesis** — target mid-September 2026

---

## Project Timeline

```
           Apr 2026     May          Jun          Jul          Aug        Sep 2026
           ──────────── ──────────── ──────────── ──────────── ────────── ──────────
T1  Lit review & knowledge base
T2  Proposal development & submission
T3  Stage 1 core: Mamba + projector
T4  Stats buffer + SWR generator
T5  Stage 1 benchmarking & ablations
T6  Stage 1 paper ──────────────────────────────────────── ◆ Paper 1 (mid Jun)
T7  Hybrid Mamba + Transformer
T8  Simulator integration & curriculum
T9  Robot deployment on Go2 Edu
T10 Real-world experiments
T11 Stage 2 paper & thesis ──────────────────────────────────────────── ◆ Paper 2 (mid Sep)
```

---

## Team

| Member | ID | Primary responsibility |
|---|---|---|
| Hasinthaka Piyumal | SE/2021/036 | Team lead, research direction, training pipeline, paper writing, WSO2 liaison |
| Praveen Dedigama | SE/2021/031 | Mamba backbone, null-space projector, hybrid stack design |
| Induwara Mihisara | SE/2021/025 | Biological grounding, data loaders, simulator integration |
| Thagya Kavindi | SE/2021/062 | Evaluation metrics, benchmark runs, ablation studies, results analysis |

**Supervisor:** Dr. Nalin Warnajith — nwarnajith@kln.ac.lk  
**Hardware:** Stage 2 hardware access (Unitree Go2 Edu at WSO2 Colombo) approved via the WSO2 Call for Proposals on Unitree Go2 Edu Robot-based Projects.

---

## Environment Requirements

Before touching a single `pip install`, read this. Getting these wrong is the most common reason setup fails completely.

### Operating system

> **Windows users: you must use WSL2.**

`mamba-ssm` and `causal-conv1d` both build CUDA C++ extensions. Their build scripts use Linux-style paths internally. On native Windows, the NVCC compiler expands those paths with backslashes inside preprocessor macros and the build crashes with `"#" not expected here` — this is an unfixed upstream bug with no workaround on Windows.

**Do all of the following steps inside WSL2 (Ubuntu 22.04 recommended).** Your Windows files are accessible at `/mnt/c/` inside WSL2 so nothing is lost.

If you have not set up WSL2 yet:
```
# In Windows PowerShell (run as Administrator)
wsl --install
# Restart, then open Ubuntu from the Start menu
```

macOS and Linux users can follow the same steps natively.

### Python version — 3.11 exactly

| Python | Status |
|---|---|
| 3.10.x | Works but not tested |
| **3.11.x** | ✅ Use this |
| 3.12+ | ❌ Broken — `distutils` was removed in 3.12; `mamba-ssm`'s CUDA builder crashes at install |
| 3.13+ | ❌ Broken for the same reason |

Check before creating your venv:
```bash
python3.11 --version   # must print Python 3.11.x
```

If 3.11 is not available in WSL:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### CUDA version

`mamba-ssm` requires a real NVIDIA GPU with CUDA 12.x. CPU-only machines cannot run training.

```bash
nvcc --version      # check CUDA compiler version
nvidia-smi          # check driver and CUDA runtime version
```

CUDA 12.1 or 12.2 are both confirmed to work.

### mamba-ssm version — 1.2.2, do not upgrade

Mamba 2 (≥ 2.0.0) eliminates the `x_proj` parameter that HippoCortex's null-space projector targets. Upgrading silently breaks projection — no error at import, only when training starts.

---

## Installation

Follow these steps in order. **Do not skip steps or reorder them.** `mamba-ssm` and `causal-conv1d` both import `torch` at build time — if torch is not already installed when you install them, the build fails immediately.

### Step 1 — Open WSL2 and navigate to the project

```bash
# In WSL2 terminal
cd /mnt/d/researches/hippo-cortex    # adjust path to match your Windows drive/folder
```

### Step 2 — Create a Python 3.11 virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt. **You must run `source .venv/bin/activate` every time you open a new WSL2 terminal.**

Verify you are using the right Python:
```bash
python --version   # must print Python 3.11.x
which python       # must point to .venv/bin/python, NOT a Windows path
```

### Step 3 — Install PyTorch with CUDA 12.1

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify before continuing:
```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
# Expected: 2.3.1+cu121 | CUDA: True
```

Do not continue to the next step if CUDA shows False.

### Step 4 — Install causal-conv1d from pre-built wheel

`causal-conv1d` must be installed from a pre-built wheel — the source build is broken on Linux when using the default GCC/CUDA combination. Download and install the wheel directly:

```bash
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.3cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

If that URL fails (cxx11abi mismatch), try the FALSE variant:
```bash
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

Verify:
```bash
python -c "import causal_conv1d; print('causal_conv1d OK')"
```

### Step 5 — Install mamba-ssm from pre-built wheel

Same approach — use the pre-built wheel:

```bash
pip install https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu122torch2.3cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

If that fails, try the FALSE variant:
```bash
pip install https://github.com/state-spaces/mamba/releases/download/v1.2.2/mamba_ssm-1.2.2+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

Verify — version must be exactly 1.2.2:
```bash
python -c "import mamba_ssm; print(mamba_ssm.__version__)"
# Expected: 1.2.2
```

### Step 6 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 7 — Install the package in editable mode

```bash
pip install -e .
```

This makes `import hippocortex` work from any directory without needing to set `PYTHONPATH`.

### Step 8 — Verify the full install

Run all three checks:

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba-ssm:', mamba_ssm.__version__)"
python -c "import hippocortex; print('hippocortex:', hippocortex.__version__)"
```

Expected output:
```
PyTorch: 2.3.1+cu121 | CUDA: True
mamba-ssm: 1.2.2
hippocortex: 0.1.0
```

### Step 9 — Log in to Weights & Biases

```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

All experiment runs log to the `hippocortex-stage1` project automatically.

---

## Download Datasets

```bash
# CIFAR-100 (~170 MB)
python scripts/download_datasets.py --dataset cifar100 --root data/cifar100

# ImageNet-R (~430 MB)
python scripts/download_datasets.py --dataset imagenet_r --root data/imagenet_r
```

> `data/` is gitignored — do not commit datasets.

---

## Running Tests

```bash
# All tests
pytest

# One module at a time
pytest tests/cl/
pytest tests/models/
pytest tests/utils/test_metrics.py

# With verbose output
pytest -v
```

Tests that require a downloaded dataset are marked `@pytest.mark.skip` by default. Remove the decorator once the dataset is present and run `pytest tests/data/ -v`.

---

## Running Experiments

### Smoke test first — always

Run this before any full experiment. It takes about 2 minutes and catches wiring bugs immediately:

```bash
python experiments/stage1_split_cifar100/run.py \
    dataset.n_tasks=2 \
    training.epochs_warmup=1 \
    training.epochs_joint=0 \
    logging.wandb_project=hippocortex-stage1-dev
```

### Full Stage 1 — Split-CIFAR100 (20 tasks)

```bash
python experiments/stage1_split_cifar100/run.py
```

Override any config value on the command line (Hydra syntax):

```bash
python experiments/stage1_split_cifar100/run.py seed=123
python experiments/stage1_split_cifar100/run.py training.batch_size=128
python experiments/stage1_split_cifar100/run.py training.epochs_warmup=2
```

### Full Stage 1 — ImageNet-R (20 tasks)

```bash
python experiments/stage1_imagenet_r/run.py
```

### Outputs

Every run writes to `results/stage1_split_cifar100/<run_name>/`:

```
acc_matrix.npy    shape (n_tasks, n_tasks) — acc_matrix[i,j] = accuracy on task j after training task i
summary.json      average accuracy, forgetting rate, BWT, memory_mb, wall_time
```

Results are also logged live on [wandb.ai](https://wandb.ai) under `hippocortex-stage1`.

---

## Config Reference

All hyperparameters live in YAML files. **Never hardcode values in Python.**

```yaml
# experiments/stage1_split_cifar100/config.yaml

seed: 42

dataset:
  name: split_cifar100
  n_tasks: 20               # sequential tasks
  n_classes_per_task: 5     # classes per task
  root: data/cifar100

model:
  mamba_d_model: 128        # hidden state dimension
  mamba_n_layers: 4         # Mamba SSM layers
  mamba_d_state: 16         # SSM state size
  cvae_latent_dim: 64       # CVAE bottleneck
  cvae_hidden_dim: 512      # CVAE MLP hidden size
  nsp_rank_budget: 200      # max singular vectors in null-space projector

training:
  epochs_warmup: 5          # Phase 1 — CE on current task only
  epochs_joint: 10          # Phase 2 — CE + ELBO + null-space projection
  epochs_consolidation: 3   # Phase 3 — sleep / consolidation
  lr: 1.0e-3
  batch_size: 64
  optimizer: adamw

logging:
  wandb_project: hippocortex-stage1
  run_name: null            # null → timestamp auto-generated
  save_dir: results/stage1_split_cifar100
```

---

## Package Layout

```
hippocortex/
  models/
    backbone.py              MambaBackbone — Mamba SSM encoder + task head     (Praveen)
    swr_generator.py         SWRGenerator  — Conditional VAE for SWR replay    (Praveen)
  cl/
    stats_buffer.py          StatsBuffer   — per-task (µ, σ²) store            (Hasinthaka)
    null_space_projector.py  NullSpaceProjector — gradient deflector P=I−UUᵀ  (Praveen)
    consolidation.py         consolidate() — sleep-phase replay loop           (Hasinthaka)
  data/
    split_cifar100.py        get_task_loaders() for Split-CIFAR100             (Induwara)
    imagenet_r.py            get_task_loaders() for ImageNet-R                 (Induwara)
  training/
    trainer.py               Trainer — 3-phase training orchestrator           (Hasinthaka)
  utils/
    seed.py                  set_seed()
    config.py                load_config()
    logging.py               get_logger(), init_wandb()
    metrics.py               average_accuracy(), average_forgetting(), BWT     (Thagya)
    memory_tracker.py        measure_memory_mb()                               (Thagya)

experiments/
  stage1_split_cifar100/    config.yaml + run.py + eval_baseline.py
  stage1_imagenet_r/        config.yaml + run.py
  stage2a_hybrid_vision/
  stage2b_sim_mujoco/
  stage2c_go2_deployment/

baselines/
  mamba-cl/                 Reference implementation (Cheng et al., 2024)
  kaggle_mamba_cl_baseline.ipynb   Kaggle notebook to reproduce baseline numbers

tests/
  cl/                       StatsBuffer, NullSpaceProjector tests
  models/                   MambaBackbone tests
  data/                     data loader tests (need dataset downloaded)
  utils/                    metrics, seed, config tests
```

---

## Troubleshooting

**`"#" not expected here` during causal-conv1d install**  
You are on native Windows, not WSL2. Source builds of `causal-conv1d` and `mamba-ssm` are broken on Windows due to a NVCC path-escaping bug. Switch to WSL2 and use the pre-built wheels in Steps 4 and 5.

**`ModuleNotFoundError: No module named 'torch'` during mamba-ssm or causal-conv1d install**  
PyTorch is not installed yet, or you are in the wrong virtual environment. Run `which python` — if the path is under `/mnt/c/` or points to a Windows `.venv`, you activated the Windows venv inside WSL2. Run `deactivate`, then `source .venv/bin/activate` from the repo root inside WSL2.

**`which python` shows a path under `/mnt/c/` inside WSL2**  
You activated the Windows `.venv` inside WSL2. They are incompatible. Deactivate and re-activate the Linux venv:
```bash
deactivate
cd /mnt/d/researches/hippo-cortex   # adjust to your path
source .venv/bin/activate
which python    # must now show /mnt/d/.../hippo-cortex/.venv/bin/python
```

**`BackendUnavailable: Cannot import 'setuptools.backends.legacy'` on `pip install -e .`**  
Your `pyproject.toml` has the wrong build backend. Open it and change:
```toml
build-backend = "setuptools.backends.legacy:build"   # wrong
```
to:
```toml
build-backend = "setuptools.build_meta"              # correct
```

**`python --version` shows 3.12 or 3.13`**  
Install Python 3.11 in WSL2:
```bash
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev
```
Then recreate the venv with `python3.11 -m venv .venv`.

**`nvcc was not found` / `bare_metal_version is not defined`**  
The CUDA Toolkit (which provides `nvcc`) is not installed. In WSL2:
```bash
sudo apt install nvidia-cuda-toolkit
```
Or install CUDA Toolkit 12.1 from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) selecting Linux → x86_64 → WSL-Ubuntu.

**`CUDA: False` when PyTorch is installed**  
You installed CPU-only PyTorch. Reinstall with the CUDA index URL from Step 3.

**`mamba_ssm.__version__` prints `2.x.x`**  
Wrong version installed. Force reinstall:
```bash
pip install mamba-ssm==1.2.2 --force-reinstall
```

**`ImportError: cannot import name 'hippocortex'`**  
Editable install is missing. From the repo root with the venv active:
```bash
pip install -e .
```

**`wandb: ERROR Not logged in`**  
```bash
wandb login   # paste API key from https://wandb.ai/authorize
```

**Tests fail with `NotImplementedError`**  
That module stub has not been implemented yet. Check the owner column in the package layout and coordinate with the responsible team member.

---

## References

- Buzsáki, G. (2015). Hippocampal sharp wave-ripple: A cognitive biomarker for episodic memory and planning. *Hippocampus*, 25(10), 1073–1188.
- Cheng, D. et al. (2024). Mamba-CL: Optimizing selective state space model in null space for continual learning. *arXiv:2411.15469*.
- De Lange, M. et al. (2022). A continual learning survey: Defying forgetting in classification tasks. *IEEE TPAMI*, 44(7), 3366–3385.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.
- Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521–3526.
- Lee, I. N. et al. (2025). Exemplar-free continual learning for state space models. *arXiv:2505.18604*.
- McClelland, J. L. et al. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419–457.
- Rebuffi, S.-A. et al. (2017). iCaRL: Incremental classifier and representation learning. *CVPR*.
- Saha, G. et al. (2021). Gradient projection memory for continual learning. *ICLR*.
- Shin, H. et al. (2017). Continual learning with deep generative replay. *NeurIPS*.
- Sohn, K. et al. (2015). Learning structured output representation using deep conditional generative models. *NeurIPS*.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS*.
