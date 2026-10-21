"""Inf-SSM baseline: class-incremental training and evaluation of Vim-small.

Follows vim/main.py + vim/cl_methods/inf_ssm.py of the official code
(https://github.com/isaacNingLee/Inf-SSM, CVPR 2026), with three additions for Kaggle:
resumable checkpoints mirrored to Hugging Face, nn.DataParallel over all visible GPUs, and
optional per-block gradient checkpointing to fit Vim-small in 16 GB.

Protocol (as in the official scripts):
- Task 0 is not trained: training starts from a Vim-small checkpoint already fine-tuned on task 0
  (`--finetune hf` downloads the authors' one, seed 0 only). `--full_run` trains task 0 too,
  e.g. from the ImageNet Vim-small weights (`--finetune vim_small_in1k`).
- Each task t >= 1: fresh AdamW + cosine schedule, 40 epochs, logits restricted to the task's
  classes, loss = soft-target CE (mixup/cutmix) + Inf-SSM penalty against the model after task t-1.
  After each epoch the current-task val accuracy is measured; the best epoch is reloaded at the end.
- The base LR is multiplied by --task_lr_scaling at every task boundary, task 0 included.
- After each task: class-incremental (and task-incremental) test accuracy on all tasks so far.
  AA = mean accuracy after the last task, AIA = mean of AA over tasks, FM = mean forgetting.
"""
import argparse
import json
import math
import os
import random
import sys
import uuid
from copy import deepcopy
from time import time as ttime

import numpy as np
import torch
import tqdm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import nn, Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from cl_datasets import build_cl_splits
from inf_ssm import InfSSMForward
from mamba_vim import KERNEL_IMPORT_ERRORS, MODELS, load_finetune

CKPT_TASK, CKPT_EPOCH, CKPT_BEST = "task_end.pt", "epoch.pt", "task_best.pt"
RESULTS, WANDB_ID = "results.json", "wandb_id.txt"
NUM_CLASSES = {'cifar100': 100, 'imagenet_r': 200, 'caltech256': 250}

# Hyperparameters of the official launch scripts (vim/scripts/*.sh = paper Tables 12/13).
# Common to all: batch 128, 40 epochs, weight decay 0.1, drop path 0.
PRESETS = {
    ('cifar100', 10): dict(lr=1e-5, warmup_lr=1e-6, min_lr=1e-7, task_lr_scaling=0.5, reg_lambda=3e4, hf_dir='cifar100_10task'),
    ('cifar100', 5): dict(lr=1e-5, warmup_lr=1e-6, min_lr=1e-7, task_lr_scaling=0.5, reg_lambda=2e5, hf_dir='cifar100_5task'),
    ('imagenet_r', 10): dict(lr=5e-4, warmup_lr=1e-4, min_lr=1e-5, task_lr_scaling=0.25, reg_lambda=2.5e5, hf_dir='10task'),
    ('imagenet_r', 5): dict(lr=5e-4, warmup_lr=1e-4, min_lr=1e-5, task_lr_scaling=0.25, reg_lambda=1e6, hf_dir='5task'),
    ('caltech256', 10): dict(lr=1e-4, warmup_lr=1e-4, min_lr=1e-5, task_lr_scaling=0.5, reg_lambda=2.5e6, hf_dir='caltech256_10task'),
    ('caltech256', 5): dict(lr=1e-4, warmup_lr=1e-4, min_lr=1e-5, task_lr_scaling=0.5, reg_lambda=2.5e6, hf_dir='caltech256_5task'),
}
TASK0_HF_REPO = "ileening0022/Inf-SSM"  # the authors' task-0 checkpoints: <hf_dir>/task_0_checkpoint_seed_<seed>.pth
VIM_SMALL_IN1K = ("hustvl/Vim-small-midclstok", "vim_s_midclstok_80p5acc.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CPU only for smoke tests


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    p = argparse.ArgumentParser(description='Inf-SSM class-incremental learning on Vim')
    p.add_argument('-d', '--dataset', required=True, choices=tuple(NUM_CLASSES))
    p.add_argument('-dr', '--data_root', default='', help='CIFAR-100: dir holding cifar-100-python/ (fetched from the team Drive if missing); ImageNet-R / Caltech-256: the extracted image folder')
    p.add_argument('-t', '--num_tasks', type=int, default=10)
    p.add_argument('--random_class', action='store_true', help='shuffle the class order with --seed (official default: natural order)')
    p.add_argument('--seed', type=int, default=0, help='paper: 0, 10, 100')
    p.add_argument('-m', '--model', default='vim_small', choices=tuple(MODELS))
    p.add_argument('--depth', type=int, default=None, help='override the number of Vim blocks (smoke tests only)')
    p.add_argument('--finetune', default='hf', help="task-0 checkpoint path; 'hf' = the authors' one for this benchmark and seed; "
                                                   "'vim_small_in1k' = ImageNet Vim-small weights (use with --full_run); 'none' = random init")
    p.add_argument('--full_run', action='store_true', help='also train task 0 (with --cl_method seq semantics, no old model)')
    p.add_argument('--cl_method', default='reg', choices=('reg', 'seq'), help='reg = Inf-SSM, seq = plain sequential fine-tuning')
    p.add_argument('--reg_mode', default='inf_ssm', choices=('inf_ssm', 'mse'))
    p.add_argument('--reg_lambda', type=float, default=None, help='default: the paper value for this benchmark')
    # optimisation (None = the paper value for this benchmark)
    p.add_argument('-e', '--epochs', type=int, default=40)
    p.add_argument('-b', '--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--warmup_lr', type=float, default=None)
    p.add_argument('--min_lr', type=float, default=None)
    p.add_argument('--task_lr_scaling', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--opt', default='adamw')
    p.add_argument('--opt_eps', type=float, default=1e-8)
    p.add_argument('--clip_grad', type=float, default=None)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--cooldown_epochs', type=int, default=10)
    p.add_argument('--decay_rate', type=float, default=0.1)
    p.add_argument('--drop_path', type=float, default=0.0)
    # augmentation (DeiT/Vim defaults)
    p.add_argument('--input_size', type=int, default=224)
    p.add_argument('--eval_crop_ratio', type=float, default=0.875)
    p.add_argument('--color_jitter', type=float, default=0.3)
    p.add_argument('--aa', default='rand-m9-mstd0.5-inc1')
    p.add_argument('--smoothing', type=float, default=0.1)
    p.add_argument('--train_interpolation', default='bicubic')
    p.add_argument('--reprob', type=float, default=0.25)
    p.add_argument('--remode', default='pixel')
    p.add_argument('--recount', type=int, default=1)
    p.add_argument('--mixup', type=float, default=0.8)
    p.add_argument('--cutmix', type=float, default=1.0)
    p.add_argument('--mixup_prob', type=float, default=1.0)
    p.add_argument('--mixup_switch_prob', type=float, default=0.5)
    p.add_argument('--mixup_mode', default='batch')
    # hardware
    p.add_argument('-jt', '--workers', type=int, default=8)
    p.add_argument('--use_amp', type=str2bool, default=False, help='fp16 autocast; the paper runs in fp32')
    p.add_argument('--data_parallel', type=str2bool, default=False, help='split each batch across all visible GPUs (nn.DataParallel)')
    p.add_argument('--grad_ckpt', type=str2bool, default=False, help='recompute each Vim block in backward: much less memory, ~30%% slower')
    # logging and checkpoints
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', default='inf-ssm-baseline')
    p.add_argument('--progress_interval', type=float, default=0, help='seconds between tqdm progress updates; 0 disables the bars')
    p.add_argument('--ckpt_dir', default='', help='resumable checkpoints and results.json under <ckpt_dir>/<run_name>; empty disables')
    p.add_argument('--hf_repo', default='', help='Hugging Face repo id to mirror the checkpoints to (token from HF_TOKEN); needs --ckpt_dir')
    p.add_argument('--hf_sync_every', type=int, default=5, help='upload the mid-task checkpoint every N epochs (0 = only at task ends)')
    p.add_argument('--hf_squash', type=str2bool, default=False, help='squash the HF repo history after each task so storage keeps one copy per file')
    args = p.parse_args()

    preset = PRESETS.get((args.dataset, args.num_tasks), {})
    for k in ('lr', 'warmup_lr', 'min_lr', 'task_lr_scaling', 'reg_lambda'):
        if getattr(args, k) is None:
            assert k in preset, f"no paper preset for {args.dataset} with {args.num_tasks} tasks: pass --{k}"
            setattr(args, k, preset[k])
    args.hf_dir = preset.get('hf_dir')
    if args.aa is not None and args.aa.lower() == 'none':
        args.aa = None
    args.nb_classes = NUM_CLASSES[args.dataset]
    args.n_classes_per_task = args.nb_classes // args.num_tasks
    if args.mixup > 0 or args.cutmix > 0:
        assert args.batch_size % 2 == 0, "mixup/cutmix needs an even batch size"
    return args


def run_name(args) -> str:
    lam = f"_lam{args.reg_lambda:g}" if args.cl_method == 'reg' else ""
    return f"inf_ssm_{args.cl_method}{lam}_{args.model}_{args.dataset}_t{args.num_tasks}_e{args.epochs}_seed_{args.seed}"


def task_lr(args, t: int) -> float:
    # main.py does `args.lr *= task_lr_scaling` at the start of every task (task 0 included)
    return args.lr * args.task_lr_scaling ** (t + 1) if args.task_lr_scaling < 1.0 else args.lr


def resolve_checkpoint(args) -> str:
    from huggingface_hub import hf_hub_download
    if args.finetune == 'hf':
        assert args.hf_dir, f"no released task-0 checkpoint for {args.dataset} with {args.num_tasks} tasks"
        fn = f"{args.hf_dir}/task_0_checkpoint_seed_{args.seed}.pth"
        try:
            return hf_hub_download(TASK0_HF_REPO, fn)
        except Exception as e:
            raise SystemExit(f"could not download {TASK0_HF_REPO}/{fn} ({type(e).__name__}). The authors only released seed 0; "
                             f"for other seeds use --full_run --finetune vim_small_in1k") from e
    if args.finetune == 'vim_small_in1k':
        return hf_hub_download(*VIM_SMALL_IN1K)
    return None if args.finetune == 'none' else args.finetune


# ---------------------------------------------------------------- state, RNG, checkpoint files

class Run:
    args: argparse.Namespace
    run_dir: str | None = None
    hub = None
    g: torch.Generator
    accum: dict  # official accum_result: {'class_i_acc': [...], 'task_i_acc': [...], 'mean_class_acc': [...], ...}
    start_time: float
    resume_epoch: dict | None = None


def rng_state(R: Run) -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, 'loader': R.g.get_state()}


def set_rng_state(R: Run, s: dict) -> None:
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])
    R.g.set_state(s['loader'])


def atomic_save(obj, path: str) -> None:
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def push(R: Run, filename: str, message: str) -> None:
    if R.hub is not None:
        R.hub.push(filename, message)


def acc_matrix(R: Run, n_tasks_done: int) -> list[list[float]]:
    """M[t][i] = class-incremental test accuracy on task i after training task t (i <= t)."""
    return [[R.accum[f'class_{i}_acc'][t - i] for i in range(t + 1)] for t in range(n_tasks_done)]


def forgetting(M: list[list[float]], t: int) -> float:
    """Official forgetting(): mean over tasks i < t of (best accuracy so far - accuracy now)."""
    if t == 0:
        return 0.0
    return sum(max(M[s][i] for s in range(i, t + 1)) - M[t][i] for i in range(t)) / t


def write_results(R: Run, t: int) -> None:
    n = t + 1
    M = acc_matrix(R, n)
    aa = R.accum['mean_class_acc']
    results = {
        'run_name': run_name(R.args),
        'tasks_done': n,
        'num_tasks': R.args.num_tasks,
        'finished': n == R.args.num_tasks,
        'avg_acc': aa[t],                      # AA (class-incremental), fraction
        'avg_inc_acc': sum(aa[:n]) / n,        # AIA
        'forgetting': forgetting(M, t),        # FM
        'avg_acc_per_task': aa[:n],
        'forgetting_per_task': [forgetting(M, s) for s in range(n)],
        'til_avg_acc_per_task': R.accum['mean_task_acc'][:n],
        'acc_matrix': M,
        'accum_result': R.accum,               # same layout as the official accum_result.json
        'elapsed_hours': (ttime() - R.start_time) / 3600,
        'args': vars(R.args),
    }
    if R.run_dir is not None:
        with open(os.path.join(R.run_dir, RESULTS), 'w') as f:
            json.dump(results, f, indent=2, default=str)
    print(f":: ** Results of task [{n}/{R.args.num_tasks}]: [ AA {results['avg_acc']:>6.2%} | AIA {results['avg_inc_acc']:>6.2%} | "
          f"FM {results['forgetting']:>6.2%} | TIL {results['til_avg_acc_per_task'][-1]:>6.2%} ] **", flush=True)
    print(f":: ** Time so far: {results['elapsed_hours']:.2f} h **", flush=True)
    if R.args.use_wandb:
        import wandb
        wandb.log({'task': t, 'AA': results['avg_acc'], 'AIA': results['avg_inc_acc'], 'FM': results['forgetting'],
                   'TIL_AA': results['til_avg_acc_per_task'][-1],
                   **{f'test/class_{i}_acc': M[t][i] for i in range(n)}})


def save_task_checkpoint(R: Run, model: nn.Module, t: int) -> None:
    if R.run_dir is None:
        return
    atomic_save({'tasks_done': t + 1, 'model': model.state_dict(), 'accum': R.accum,
                 'elapsed': ttime() - R.start_time, 'rng': rng_state(R)}, os.path.join(R.run_dir, CKPT_TASK))
    for fn in (CKPT_EPOCH, CKPT_BEST):
        if os.path.exists(_p := os.path.join(R.run_dir, fn)):
            os.remove(_p)
    print(f":: checkpoint saved: {R.run_dir} (tasks done: {t + 1})", flush=True)
    push(R, CKPT_TASK, f"task {t + 1} done")
    push(R, RESULTS, f"results after task {t + 1}")
    if R.hub is not None:
        R.hub.squash_history()


def setup_checkpointing(R: Run, model: nn.Module) -> int:
    """Pull this run's files from the HF mirror if they are not on disk, then restore them. Returns the number of finished tasks."""
    args = R.args
    if not args.ckpt_dir:
        assert not args.hf_repo, "--hf_repo needs --ckpt_dir"
        return 0
    R.run_dir = os.path.join(args.ckpt_dir, run_name(args))
    os.makedirs(R.run_dir, exist_ok=True)
    if args.hf_repo:
        from hf_sync import HubMirror
        R.hub = HubMirror(args.hf_repo, run_name(args), args.ckpt_dir, squash=args.hf_squash)
        if pulled := R.hub.pull([CKPT_TASK, CKPT_EPOCH, CKPT_BEST, RESULTS, WANDB_ID]):
            print(f":: pulled {pulled} from hf://{args.hf_repo}/{run_name(args)}")

    tasks_done = 0
    if os.path.exists(_p := os.path.join(R.run_dir, CKPT_TASK)):
        ckpt = torch.load(_p, map_location='cpu', weights_only=False)
        tasks_done = ckpt['tasks_done']
        model.load_state_dict(ckpt['model'])
        R.accum = ckpt['accum']
        R.start_time = ttime() - ckpt['elapsed']
        set_rng_state(R, ckpt['rng'])
        print(f":: resumed from {_p}: {tasks_done}/{args.num_tasks} tasks done", flush=True)

    if os.path.exists(_p := os.path.join(R.run_dir, CKPT_EPOCH)):
        ep = torch.load(_p, map_location='cpu', weights_only=False)
        if ep['task'] == tasks_done:  # an older one belongs to a task that already finished
            assert os.path.exists(os.path.join(R.run_dir, CKPT_BEST)), f"{CKPT_EPOCH} without {CKPT_BEST}"
            R.resume_epoch = ep
    return tasks_done


def init_wandb(R: Run) -> None:
    import wandb
    _id_path = os.path.join(R.run_dir, WANDB_ID) if R.run_dir else None
    if _id_path and os.path.exists(_id_path):
        with open(_id_path) as f:
            run_id = f.read().strip()
    else:
        run_id = uuid.uuid4().hex[:12]
    wandb.init(project=R.args.wandb_project, name=run_name(R.args), config=vars(R.args), id=run_id, resume='allow')
    if _id_path and not os.path.exists(_id_path):
        with open(_id_path, 'w') as f:
            f.write(run_id)
        push(R, WANDB_ID, "wandb run id")


# ---------------------------------------------------------------- train / eval

def gpu_mem_str() -> str:
    if DEVICE.type != 'cuda':
        return ""
    peaks = [torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(torch.cuda.device_count())]
    return " peak_mem=" + "/".join(f"{p:.1f}" for p in peaks) + "GB"


def maybe_parallel(R: Run, module: nn.Module) -> nn.Module:
    # DataParallel only wraps the forward pass: parameters, optimizer and checkpoints stay on the plain model
    if R.args.data_parallel and torch.cuda.device_count() > 1:
        return nn.DataParallel(module)
    return module


def train_one_epoch(R: Run, t: int, epoch: int, model: nn.Module, old_net: nn.Module | None, loader: DataLoader,
                    optimizer, criterion, mixup_fn, scaler) -> dict:
    args = R.args
    model.train()
    fwd = maybe_parallel(R, InfSSMForward(model, old_net, args.reg_lambda, args.reg_mode))
    pc, ac = t * args.n_classes_per_task, (t + 1) * args.n_classes_per_task
    sums = dict(loss=0., reg_loss=0., correct=0, count=0, steps=0)

    for samples, targets in tqdm.tqdm(loader, total=len(loader), desc=f"Task [{t + 1}] epoch [{epoch + 1}] train", dynamic_ncols=True,
                                      disable=args.progress_interval <= 0, mininterval=args.progress_interval):
        samples = samples.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        else:
            targets = nn.functional.one_hot(targets, args.nb_classes).float() * (1 - args.smoothing) + args.smoothing / args.nb_classes

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            outputs, penalty = fwd(samples)
            penalty = penalty.mean()
            outputs = outputs[:, pc:ac]
            targets = targets[:, pc:ac] / targets[:, pc:ac].sum(dim=1, keepdim=True)  # back to sum = 1
            loss = criterion(outputs, targets) + penalty

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training", flush=True)
            sys.exit(1)
        scaler.scale(loss).backward()
        if args.clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            sums['correct'] += (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
        sums['count'] += samples.shape[0]
        sums['loss'] += loss_value
        sums['reg_loss'] += penalty.item()
        sums['steps'] += 1

    n = max(sums['steps'], 1)
    return {'loss': sums['loss'] / n, 'reg_loss': sums['reg_loss'] / n, 'acc': sums['correct'] / max(sums['count'], 1),
            'lr': optimizer.param_groups[0]['lr']}


@torch.no_grad()
def cl_evaluate(R: Run, model: nn.Module, loader: DataLoader, task_id: int, current: bool = False) -> dict:
    """Official cl_evaluate: class-incremental accuracy over the logits of tasks 0..task_id (and task-incremental
    accuracy within each task's logits), per task. current=True evaluates only task_id (used on the val split)."""
    model.eval()
    fwd = maybe_parallel(R, model)
    cpt = R.args.n_classes_per_task
    result = {}
    for task in range(task_id if current else 0, task_id + 1):
        loader.dataset.set_task(task)
        cil, til = 0, 0
        for images, target in loader:
            images, target = images.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=R.args.use_amp):
                output = fwd(images)[:, :(task_id + 1) * cpt]
            cil += (output.argmax(dim=1) == target).sum().item()
            til += (output[:, task * cpt:(task + 1) * cpt].argmax(dim=1) + task * cpt == target).sum().item()
        result[f'class_{task}_acc'] = cil / len(loader.dataset)
        result[f'task_{task}_acc'] = til / len(loader.dataset)
    if not current:
        result['mean_class_acc'] = sum(result[f'class_{i}_acc'] for i in range(task_id + 1)) / (task_id + 1)
        result['mean_task_acc'] = sum(result[f'task_{i}_acc'] for i in range(task_id + 1)) / (task_id + 1)
    return result


def train_one_task(R: Run, t: int, model: nn.Module, loader_train: DataLoader, loader_val: DataLoader, criterion, mixup_fn) -> None:
    args = R.args
    old_net = None
    if t > 0 and args.cl_method == 'reg':
        old_net = deepcopy(model).eval()  # the model after task t-1
        for p in old_net.parameters():
            p.requires_grad = False

    lr = task_lr(args, t)
    optimizer = create_optimizer_v2(model, opt=args.opt, lr=lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    scheduler, _ = create_scheduler_v2(optimizer, sched='cosine', num_epochs=args.epochs, min_lr=args.min_lr, warmup_lr=args.warmup_lr,
                                       warmup_epochs=args.warmup_epochs, cooldown_epochs=args.cooldown_epochs, decay_rate=args.decay_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and DEVICE.type == 'cuda')
    loader_train.dataset.set_task(t)

    best_acc, best_epoch, start_epoch = -1.0, -1, 0  # official starts at 0; -1 just guarantees a saved best model
    best_path = os.path.join(R.run_dir, CKPT_BEST) if R.run_dir else None
    best_state = None  # used when checkpointing is off
    if (ep := R.resume_epoch) is not None and ep['task'] == t:
        model.load_state_dict(ep['model'])
        optimizer.load_state_dict(ep['optimizer'])
        if ep.get('scaler') is not None:
            scaler.load_state_dict(ep['scaler'])
        scheduler.step(ep['epoch'])
        set_rng_state(R, ep['rng'])
        best_acc, best_epoch, start_epoch = ep['best_acc'], ep['best_epoch'], ep['epoch'] + 1
        print(f"Task [{t + 1}/{args.num_tasks}]:: resumed after epoch {start_epoch} (best val {best_acc:.2%} at epoch {best_epoch + 1})", flush=True)
    R.resume_epoch = None
    print(f"Task [{t + 1}/{args.num_tasks}]:: {len(loader_train.dataset)} train images, base lr {lr:.3g}, "
          f"{'Inf-SSM ' + args.reg_mode + f' lambda={args.reg_lambda:g}' if old_net is not None else 'no regularizer'}", flush=True)

    _ttimer = ttime()
    for epoch in range(start_epoch, args.epochs):
        _etimer = ttime()
        stats = train_one_epoch(R, t, epoch, model, old_net, loader_train, optimizer, criterion, mixup_fn, scaler)
        scheduler.step(epoch)
        val_acc = cl_evaluate(R, model, loader_val, t, current=True)[f'class_{t}_acc']
        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch
            if best_path:
                atomic_save({'task': t, 'epoch': epoch, 'val_acc': val_acc, 'model': model.state_dict()}, best_path)
            else:
                best_state = deepcopy(model.state_dict())
        print(f"Task [{t + 1}/{args.num_tasks}] Epoch [{epoch + 1:>2}/{args.epochs}]:: loss={stats['loss']:.4f} reg={stats['reg_loss']:.4f} "
              f"train_acc={stats['acc']:6.2%} lr={stats['lr']:.3g} val_acc={val_acc:6.2%}{' *' if improved else ''} "
              f"time={ttime() - _etimer:.0f}s{gpu_mem_str()}", flush=True)
        if args.use_wandb:
            import wandb
            wandb.log({'task': t, 'epoch': epoch, 'train/loss': stats['loss'], 'train/reg_loss': stats['reg_loss'],
                       'train/acc': stats['acc'], 'lr': stats['lr'], 'val/acc': val_acc})
        if R.run_dir:
            atomic_save({'task': t, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict() if args.use_amp else None, 'best_acc': best_acc, 'best_epoch': best_epoch,
                         'rng': rng_state(R)}, os.path.join(R.run_dir, CKPT_EPOCH))
            if args.hf_sync_every > 0 and (epoch + 1) % args.hf_sync_every == 0 and epoch + 1 < args.epochs:
                push(R, CKPT_BEST, f"task {t + 1} best model")  # before epoch.pt, which refers to it
                push(R, CKPT_EPOCH, f"task {t + 1} epoch {epoch + 1}")

    best = torch.load(best_path, map_location='cpu', weights_only=False)['model'] if best_path else best_state
    model.load_state_dict(best)
    print(f"Task [{t + 1}/{args.num_tasks}]:: loaded best model from epoch {best_epoch + 1} (val {best_acc:.2%}); "
          f"training time {(ttime() - _ttimer) / 60:.1f} min", flush=True)


def main():
    args = get_args()
    R = Run()
    R.args = args
    R.accum = {}
    R.start_time = ttime()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    R.g = torch.Generator()
    R.g.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if DEVICE.type == 'cuda' and KERNEL_IMPORT_ERRORS:
        # the PyTorch reference scan is ~10x slower and needs several GB per layer: never train with it on a GPU
        raise SystemExit(f"mamba CUDA kernels unavailable: {KERNEL_IMPORT_ERRORS}")
    print(f":: run {run_name(args)}")
    print(f":: lr={args.lr:g} warmup_lr={args.warmup_lr:g} min_lr={args.min_lr:g} task_lr_scaling={args.task_lr_scaling:g} "
          f"reg_lambda={args.reg_lambda:g} batch={args.batch_size} epochs={args.epochs} amp={args.use_amp} grad_ckpt={args.grad_ckpt}")
    if args.data_parallel:
        print(f":: data_parallel: batches split across {torch.cuda.device_count()} GPU(s)")

    partition_path = None
    if args.ckpt_dir:
        os.makedirs(_rd := os.path.join(args.ckpt_dir, run_name(args)), exist_ok=True)
        partition_path = os.path.join(_rd, 'partition.json')
    ds_train, ds_val, ds_test = build_cl_splits(args, R.g, partition_path)
    loader_kw = dict(num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=R.g)
    loader_train = DataLoader(ds_train, sampler=RandomSampler(ds_train), batch_size=args.batch_size, drop_last=True, **loader_kw)
    loader_val = DataLoader(ds_val, sampler=SequentialSampler(ds_val), batch_size=args.batch_size * 2, drop_last=False, **loader_kw)
    loader_test = DataLoader(ds_test, sampler=SequentialSampler(ds_test), batch_size=args.batch_size * 2, drop_last=False, **loader_kw)

    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
                         mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.nb_classes)
    criterion = SoftTargetCrossEntropy()

    model = MODELS[args.model](num_classes=args.nb_classes, drop_path_rate=args.drop_path, img_size=args.input_size, grad_ckpt=args.grad_ckpt,
                               **({'depth': args.depth} if args.depth else {}))
    if (ckpt_path := resolve_checkpoint(args)) is not None:
        load_finetune(model, ckpt_path)
    else:
        assert args.full_run, "--finetune none starts from random weights, which needs --full_run"
    model.to(DEVICE)
    print(f":: {args.model}: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M trainable params")

    tasks_done = setup_checkpointing(R, model)
    if tasks_done == args.num_tasks:
        print(f":: all {tasks_done} tasks already done; results in {R.run_dir}/{RESULTS}")
    if args.use_wandb:
        init_wandb(R)

    for t in range(tasks_done, args.num_tasks):
        if t > 0 or args.full_run:
            train_one_task(R, t, model, loader_train, loader_val, criterion, mixup_fn)
        result = cl_evaluate(R, model, loader_test, t)
        for k, v in result.items():
            R.accum.setdefault(k, []).append(v)
        print(f"Task [{t + 1}/{args.num_tasks}]:: test " + " ".join(f"T{i + 1}={result[f'class_{i}_acc']:.2%}" for i in range(t + 1)), flush=True)
        write_results(R, t)
        save_task_checkpoint(R, model, t)

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
