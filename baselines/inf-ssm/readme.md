# Inf-SSM baseline (Vim-small)

Port of the official code for **Exemplar-Free Continual Learning for State Space Models**
(Lee, Mahmoodi, Le, Harandi, CVPR 2026): https://github.com/isaacNingLee/Inf-SSM.
It follows the official `vim/main.py`, `vim/cl_methods/inf_ssm.py` and `vim/datasets.py`, and adds the
Kaggle features we use for Mamba-CL: resumable checkpoints mirrored to Hugging Face, `nn.DataParallel`
across GPUs, and optional gradient checkpointing.

Kaggle notebook: [`notebooks/kaggle/2026-10-21_inf_ssm_vim_small_cifar100.ipynb`](../../notebooks/kaggle/2026-10-21_inf_ssm_vim_small_cifar100.ipynb).
Sequential fine-tuning (no CL method, the lower bound), same protocol and start:
[`notebooks/kaggle/2026-11-02_seq_finetune_vim_small_cifar100.ipynb`](../../notebooks/kaggle/2026-11-02_seq_finetune_vim_small_cifar100.ipynb).
Mamba-CL under this same protocol:
[`notebooks/kaggle/2026-11-02_mamba_cl_infssm_protocol_cifar100.ipynb`](../../notebooks/kaggle/2026-11-02_mamba_cl_infssm_protocol_cifar100.ipynb).

| File | What |
|---|---|
| `train_eval.py` | Class-incremental training + evaluation, checkpoints, HF mirror, W&B |
| `mamba_vim.py` | Vim-small/tiny and the bidirectional Mamba mixer that exposes the per-token `(A, C)` states |
| `inf_ssm.py` | Inf-SSM distance on the infinite Grassmannian, penalty, DataParallel-safe forward wrapper |
| `mamba_cl.py` | Mamba-CL's null-space projection as `--cl_method mamba_cl` (model from `../mamba-cl/utils/vim_mamba.py`) |
| `cl_datasets.py` | Split CIFAR-100 / ImageNet-R / Caltech-256 with the official splits and transforms |
| `hf_sync.py` | Hugging Face mirror (same as Mamba-CL) |
| `run_benchmark.sh` | Paper runs for one benchmark over seeds 0, 10, 100 |

## Method in one paragraph

Every token of every Vim layer and scan direction is a diagonal SSM with
`A = SN(mean_d exp(softplus(delta) * A_cont))` and `C = SN(C)` in R^16 (`SN(x) = 2 sigmoid(x) - 1`).
For the current and the frozen previous-task model, the extended observability Gramian of the stacked
system is solved in closed form, `P_ij = c_i c_j / (1 - a_i a_j)`, and the penalty is
`lambda * mean(1 - tr(P12 P21) / (tr(P11) tr(P22)))` over tokens, layers and directions. No exemplars.

## Environment

Same as Mamba-CL: Python 3.11, torch 2.3.1 + cu121, `causal-conv1d==1.4.0`, `mamba-ssm==1.2.2`,
plus `timm==1.0.9 einops numpy==1.26.4 transformers==4.41.2 gdown huggingface_hub wandb`.
`transformers` must stay at 4.41.2: mamba-ssm 1.2.2 imports it, and newer releases refuse torch 2.3.1, which
makes the mamba import fail and the model fall back to the slow PyTorch scan.

CIFAR-100 is fetched from the team's Google Drive copy (`scripts/download_datasets.py`, same as Mamba-CL)
into `<data_root>/cifar-100-python` when it is missing.

The official code needs its own fork of mamba-ssm 1.1.1 (fused Vim kernels). `mamba_vim.py` builds the same
computation from the stock `selective_scan_fn` and `causal_conv1d_fn`, so no fork is compiled. Without
those kernels (or on CPU) it falls back to a pure-PyTorch scan, which is only fast enough for tests.

## Running

```bash
cd baselines/inf-ssm
# paper settings for CIFAR-100 / 10 tasks, seed 0, from the authors' task-0 checkpoint
python train_eval.py -d cifar100 -dr ../../data/cifar100 -t 10 --seed 0 --finetune hf \
    --ckpt_dir ./checkpoints --data_parallel true --grad_ckpt true
# all three paper seeds
bash run_benchmark.sh cifar100 10
# sequential fine-tuning lower bound, seed 0
CL_METHOD=seq bash run_benchmark.sh cifar100 10 0
```

`-d cifar100 | imagenet_r | caltech256`, `-t 5 | 10` pick the paper hyperparameters (`PRESETS` in
`train_eval.py`, from the official `vim/scripts/*.sh` = paper Tables 12/13). Any of `--lr --warmup_lr
--min_lr --task_lr_scaling --reg_lambda` overrides them. `--cl_method seq` is the sequential fine-tuning
lower bound; `--reg_mode mse` is the plain-MSE ablation.

### Mamba-CL under this protocol (`--cl_method mamba_cl`)

Mamba-CL's own protocol (separate head per task, Adam lr 2e-4 with a x50 head LR, AutoAugment, last epoch)
differs from this one in almost everything but the method, so its numbers cannot be put next to Inf-SSM's.
`--cl_method mamba_cl` keeps Mamba-CL's method and takes everything else from this trainer:

- model: `../mamba-cl/utils/vim_mamba.py` (same weights and computation as `mamba_vim.py`, `x_proj` split
  into `_step/_B/_C`, IntermReaders on the inputs of the projected parameters);
- trained: the head plus `x_proj_{step,B,C}`, `A_log`, `A_b_log`, `out_proj.weight` of every block (Mamba-CL's
  `--training_string`); the rest of the backbone is frozen;
- after each task, the inputs of those 216 parameters are collected over the task's train images (eval
  transform), summed over tasks, and Mamba-CL's adaptive SVD threshold gives the null space;
- every AdamW step of a projected parameter is replaced by its projection `eta * P_null + (1 - eta) * I`
  (`--null_eta`, default 0.95 as in Mamba-CL for CIFAR-100). The head is not projected.

`task_end.pt` then also stores the summed covariances (~0.5 GB for Vim-small). Needs `scipy`.

`--optim mamba_cl` also takes Mamba-CL's optimisation: Adam lr 2e-4 (`--lr` overrides), head LR x50
(`--head_lr_scale`), weight decay 5e-5, no per-task LR scaling, no warmup, LR x0.1 at 50% / 80% of the epochs.
Use it with `--cl_method mamba_cl`. With the Inf-SSM optimiser (1e-6 falling to 1e-7, made for training all
weights), Mamba-CL's few trainable parameters do not move. Our first CIFAR-100 run stayed at T1 ≈ 97% and every
later task near 0% (AA heading to ~10%). The run name gets `_mclopt_lr<lr>`.

```bash
CL_METHOD=mamba_cl EXTRA="--optim mamba_cl" bash run_benchmark.sh cifar100 10 0
```

Task 0 is never trained in the paper runs: they start from a Vim-small already fine-tuned on task 0.
`--finetune hf` downloads the authors' checkpoint from
[`ileening0022/Inf-SSM`](https://huggingface.co/ileening0022/Inf-SSM). **Only seed 0 is released.** For
seeds 10 and 100 use `--full_run --finetune vim_small_in1k`: task 0 is then trained from the ImageNet
Vim-small weights with the same settings. (The authors' CIFAR task-0 checkpoint was trained that way, but
with `warmup_lr = 5e-6` instead of `1e-6`, according to the args stored in it.)

Results go to `<ckpt_dir>/<run_name>/results.json`: `avg_acc` (AA), `avg_inc_acc` (AIA), `forgetting`
(FM), the accuracy matrix and the official `accum_result` layout. All accuracies are class-incremental.

### Checkpoints and resuming

With `--ckpt_dir`, the run writes `task_end.pt` (model + results after each task), `epoch.pt` (model,
optimizer, RNG after each epoch) and `task_best.pt` (best-val model of the current task). Re-running the
same command continues from where it stopped, also mid-task, with the same RNG streams. `--hf_repo` mirrors
them to a private HF repo (`HF_TOKEN`), pulling them back on a fresh machine; mid-task checkpoints are
uploaded every `--hf_sync_every` epochs.

## Paper numbers (Table 2, Vim-small, mean ± std over seeds 0/10/100)

| | ImageNet-R AA / AIA / FM | CIFAR-100 AA / AIA / FM | Caltech-256 AA / AIA / FM |
|---|---|---|---|
| 5 tasks | 49.34 / 67.51 / 25.14 | 45.18 / 67.34 / 36.59 | 50.75 / 67.04 / 49.93 |
| 10 tasks | 43.82 / 62.82 / 36.34 | 26.53 / 54.24 / 24.00 | 39.88 / 62.28 / 55.85 |

## Behaviour kept from the official code on purpose

These look odd but change the numbers, so they are reproduced as-is:

- The base LR is multiplied by `task_lr_scaling` at **every** task, task 0 included (task t uses
  `lr * s^(t+1)`); warmup and minimum LR are not scaled.
- `A_bar` uses `softplus(delta)` **without** `dt_proj.bias` (the scan kernel itself adds it).
- The distance returns exactly 0 only through an `allclose` shortcut (first step of a task); for
  slightly different systems it starts from `1 - ||P11||^2 / tr(P11)^2 > 0`, not from 0.
- CIFAR-100 val (10% of train) uses the train augmentation, and train images are normalised with
  ImageNet statistics while test images use CIFAR statistics.
- Splits and class order do not depend on `--seed` (a fixed seed-0 generator, classes in natural order).
- The best epoch per task is chosen by current-task val accuracy and reloaded at the end of the task.

Left out: the model EMA of the official script (updated and saved but never evaluated), t-SNE dumps,
MLflow logging, and the unused DeiT/distillation options.
