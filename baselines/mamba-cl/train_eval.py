import argparse
import json
import os
import random
import uuid
from collections import OrderedDict
import tqdm
from typing import Any, Literal
from copy import deepcopy
from time import time as ttime
import inspect
import scipy.ndimage
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn, Tensor
from torch.cuda.amp.grad_scaler import GradScaler
import torch.utils.hooks
from torch.utils.data import DataLoader, TensorDataset
import torch.linalg
import torchvision
from einops import rearrange
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from utils.mod_adam import ModAdam
from utils.config import build_model, get_config
from utils.dataset_builder import ImagePathDatasetClassManager, ImagePathDataset, define_dataset
from utils.continual_manager import ClassIncrementalManager
from utils import misc

import utils.mamba_block
import utils.defocus_attention_network
from utils.defocus_attention_network import DefocusAttentionNetwork

torch.set_float32_matmul_precision("high")


class GlobalVarsManager:
    args: argparse.Namespace
    path_data_dict: dict[str, ImagePathDataset]
    cl_mngr: ClassIncrementalManager
    acc_mat_dict: OrderedDict[str, np.ndarray]
    cache_dict: dict
    param_dict: dict[Literal['base_params', 'task_params_'], OrderedDict[str, Tensor]]
    label_map_g2l: dict[int, tuple[int, int, int]]  # {original_label: (taskid, local_label, global_label)}
    run_dir: str | None  # <ckpt_dir>/<run_name>; None when checkpointing is off
    hub: Any  # utils.hf_sync.HubMirror or None

    def init_from_args(self, args, this_file=__file__):
        self.args = args
        self.run_dir = None
        self.hub = None
        _dataset_class_manager = ImagePathDatasetClassManager(**{args.dataset: args.data_root})
        self.path_data_dict = {'train': _dataset_class_manager[args.dataset](train=True),
                               'eval': _dataset_class_manager[args.dataset](train=False)}
        self.cl_mngr = ClassIncrementalManager(self.path_data_dict['eval'].class_list, args.num_tasks, args.seed, shuffle=args.shuffle_classes)
        self.acc_mat_dict = OrderedDict(AccClassIncMat=np.zeros([_nt := self.cl_mngr.num_tasks, _nt]), AccClassIncList=np.zeros([_nt]))
        self.cache_dict = {}
        self.param_dict = {}
        self.label_map_g2l = {}

    def update_label_maps(self, taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
        _g2l_map = misc.make_label_maps(taskid, task_classes)
        if not all([_k not in self.label_map_g2l.keys() for _k in _g2l_map.keys()]):
            print("The global_to_local label map has been fully loaded, which is not expected.")
        self.label_map_g2l.update(_g2l_map)
        return _g2l_map


def get_args():
    parser = argparse.ArgumentParser(description='Class-incremental Learning')
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('cifar100', 'imagenet_r', 'sdomainet'), help='use lowercase')
    parser.add_argument('-dr', '--data_root', type=str, default="")
    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 5, 10, 20, 25, 50, 100))
    parser.add_argument('--shuffle_classes', type=misc.str2bool, default=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('-m', '--model', type=str, default='defocus_mamba_large', choices=('defocus_mamba_large', 'defocus_mamba_base', 'vim_small'),
                        help='backbone; De-focus config read from utils/<model>.yaml (large uses the 22k config); vim_small = utils/vim_mamba.py')
    parser.add_argument('--pretrained_path', type=str, default="../datasets/defocus_mamba_large_cls_21k.pth", help="checkpoint file; for vim_small also 'vim_small_in1k' (hustvl ImageNet weights from HF)")
    parser.add_argument('--grad_ckpt', type=misc.str2bool, default=False, help='vim_small only: recompute each block in backward (less memory, slower)')
    parser.add_argument('--head_dim_type', type=str, choices=('task_classes', 'pretrained', 'text_dim'), default='task_classes')
    parser.add_argument('--logit_type', type=str, choices=('head_out', 'sim_imgtext'), default='head_out')
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)
    parser.add_argument('--pretrained_exp', type=str, default="")
    parser.add_argument('--pretrained_ignore_patterns', type=str, nargs="*", default=[])
    parser.add_argument('--use_null_space', action='store_true')
    parser.add_argument('--null_patterns', type=str, nargs='+', default=('x_proj', 'out_proj.weight', 'A_log'))
    parser.add_argument('--null_eta', type=float, default=1.)
    parser.add_argument('--null_interm_accum', type=str, choices=('sum', 'mean'), default='sum')
    parser.add_argument('--refine_head', type=misc.str2bool, default=False)
    parser.add_argument('--rand_conv', type=misc.str2bool, default=True)
    parser.add_argument('--refine_epochs', type=int, default=20)
    parser.add_argument('--transform_type', type=str, choices=('timm', 'autoaug',), default='autoaug')
    parser.add_argument('--interp_mode', type=str, choices=('auto', 'bilinear', 'bicubic'), default='auto')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=200)
    parser.add_argument('-jt', '--workers', type=int, default=16)
    parser.add_argument('-je', '--eval_workers', type=int, default=2)
    parser.add_argument('-et', '--expand_times', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--use_amp', type=misc.str2bool, default=False)
    parser.add_argument('--data_parallel', type=misc.str2bool, default=False, help='split each train/eval batch across all visible GPUs (nn.DataParallel); null-space collection stays on one GPU')
    parser.add_argument('--sample_type', type=str, choices=('path', 'image'), default='image')
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--persistent_workers', type=misc.str2bool, default=False)
    parser.add_argument('--training_string', type=str, nargs='+', default=('head', 'x_proj', 'out_proj.weight', 'A_log'))
    parser.add_argument('-eb', '--eval_batch_size', type=int, default=100)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.0002)
    parser.add_argument('--lr_scale', type=float, default=50)
    parser.add_argument('--lr_scale_patterns', type=str, nargs='+', default=('head',))
    parser.add_argument('--optimizer', type=str, default='mod_adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('-dm', '--decay_milestones', type=int, nargs='+', default=[5, 8])
    parser.add_argument('--decay_epochs', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--progress_interval', type=float, default=0, help='seconds between tqdm progress updates; 0 disables the bars')
    parser.add_argument('--ckpt_dir', type=str, default="", help='save resumable checkpoints and results.json under <ckpt_dir>/<run_name>; empty disables')
    parser.add_argument('--hf_repo', type=str, default="", help='Hugging Face repo id to mirror the checkpoints to (token from HF_TOKEN); needs --ckpt_dir')
    parser.add_argument('--hf_sync_epochs', type=misc.str2bool, default=True, help='also upload the per-epoch checkpoint, not only the per-task one')
    parser.add_argument('--hf_squash', type=misc.str2bool, default=False, help='squash the HF repo history after each task so storage keeps one copy per file')
    args = parser.parse_args()

    if args.interp_mode == 'auto':
        match args.dataset:
            case 'imagenet_r' | 'sdomainet':
                args.interp_mode = 'bilinear'
            case 'cifar100':
                args.interp_mode = 'bicubic'

    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    try:
        torchvision.set_image_backend('accimage')
    except Exception:
        torchvision.set_image_backend('PIL')  # accimage not available; PIL is functionally identical


def set_model_mode(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, training: bool, to_gpu: bool = True, training_string: tuple[str] = ('',)) -> DefocusAttentionNetwork:
    for n, p in model.named_parameters():
        if training and any([_s in n for _s in training_string]):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    params_requires_grad = [n for n, p in model.named_parameters() if p.requires_grad]

    model.eval()
    for n, m in model.named_modules():
        if training and any([n.endswith(_s) and not isinstance(m, nn.Identity) for _s in training_string]):
            m.train()
        else:
            m.eval()
    modules_training = [n for n, m in model.named_modules() if m.training]

    if to_gpu:
        model.cuda()

    if training:
        pass
    else:
        assert len(params_requires_grad) == 0, f"{params_requires_grad}"
        assert len(modules_training) == 0, f"{modules_training}"

    return model


def set_learning_rates(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, base_lr: float, lr_scale: float, lr_scale_patterns: str) -> list[dict[str: Tensor | float]]:
    param_lr_groups = [{'params': [], 'lr': base_lr},
                       {'params': [], 'lr': base_lr * lr_scale}]
    lr_param_dict = {_p['lr']: [] for _p in param_lr_groups}

    for n, p in model.named_parameters():
        if p.requires_grad:
            _group_idx = 1 if any(_s in n for _s in lr_scale_patterns) else 0
            param_lr_groups[_group_idx]['params'].append(p)
            lr_param_dict[param_lr_groups[_group_idx]['lr']].append(n)

    return param_lr_groups


def forward_module(GVM: GlobalVarsManager, model: DefocusAttentionNetwork) -> nn.Module:
    # DataParallel only wraps the forward pass: parameters, optimizer and checkpoints stay on the plain model
    if GVM.args.data_parallel and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def train_one_epoch(GVM: GlobalVarsManager, curr_epoch: int, dataloader: DataLoader, model: DefocusAttentionNetwork, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer) -> str:
    args = GVM.args
    temperature: float = args.temperature
    use_amp: bool = args.use_amp
    assert temperature > 0.

    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%")
    _btimer = ttime()
    fwd = forward_module(GVM, model)

    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), desc=f"Task [{GVM.cl_mngr.current_taskid + 1}] epoch [{curr_epoch}] train", dynamic_ncols=True, disable=args.progress_interval <= 0, mininterval=args.progress_interval):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits: Tensor = fwd(images)

        if i_batch == 1:
            if args.seperate_head:
                assert logits.shape[1] == len(GVM.cl_mngr.current_task_classes)
            else:
                assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes)

        loss: Tensor = criterion(logits / temperature, target)

        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()
        amp_scalar.step(optimizer)
        amp_scalar.update()

        acc_top1, = misc.calc_accuracy(logits, target, topk=(1,))
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), loss=loss.item(), batch_time=batch_time, acc_top1=acc_top1)
        _btimer = ttime()

    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())
    return _epoch_scalar_str


def cache_state(GVM: GlobalVarsManager, taskid: int, epoch: int, model: DefocusAttentionNetwork, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler):

    task_params = OrderedDict()

    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone().cpu()

    assert not f'task_params_{taskid}' in GVM.param_dict
    GVM.param_dict[f'task_params_{taskid}'] = dict(filter(lambda n_p: 'head' in n_p[0], task_params.items()))


def train_one_task(GVM: GlobalVarsManager, taskid: int, task_classes: list[int], model: DefocusAttentionNetwork, **kwargs) -> DefocusAttentionNetwork:
    args = GVM.args
    if args.epochs == 0:
        return model

    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks)

    model: DefocusAttentionNetwork = set_model_mode(GVM, model, training=True, training_string=GVM.cache_dict['training_string'])
    model = modify_head(GVM, model, training=True, task_classes=task_classes)

    dataset = define_dataset(GVM, task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, expand_times=args.expand_times)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, timeout=args.timeout if args.workers > 0 else 0,
                            drop_last=False, persistent_workers=args.persistent_workers)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.lr_scale == 1:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())
    else:
        param_groups = set_learning_rates(GVM, model, args.lr, args.lr_scale, args.lr_scale_patterns)

    if taskid == 0:
        GVM.cache_dict['update_proj_dict'] = {}
    if args.use_null_space:
        if taskid == 0:
            GVM.cache_dict['null_param_id_dict'] = get_param_id_dict(model, args.null_patterns)
            GVM.cache_dict['interm_tensor_dict'] = {}
        else:
            assert sorted(list(GVM.cache_dict['update_proj_dict'].keys())) == sorted(list(GVM.cache_dict['null_param_id_dict'].keys()))
    else:
        assert GVM.cache_dict['update_proj_dict'] == {}

    if args.optimizer == 'mod_adam':
        optimizer = ModAdam(param_groups, update_projection_dict=GVM.cache_dict['update_proj_dict'], arg_dict={}, lr=args.lr, weight_decay=args.weight_decay, foreach=True)
    else:
        optimizer = create_optimizer_v2(param_groups, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, foreach=True)

    scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=args.epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
    assert num_epochs == args.epochs

    start_epoch = 0
    if (_resume := GVM.cache_dict.pop('resume_epoch_state', None)) is not None and _resume['task'] == taskid:
        load_model_state(model, _resume['model'], strict=False)
        optimizer.load_state_dict(_resume['optimizer'])
        set_rng_state(_resume['rng'])
        start_epoch = _resume['epoch']
        print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: resumed after epoch {start_epoch}", flush=True)

    torch.cuda.empty_cache()
    for epoch in range(0, args.epochs + 1):
        if epoch > start_epoch:
            _epoch_scalar_str = train_one_epoch(GVM, epoch, dataloader, model, criterion, optimizer)
            print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr := str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
        scheduler.step(epoch)
        if epoch > start_epoch:
            save_epoch_checkpoint(GVM, model, optimizer, taskid, epoch)

    cache_state(GVM, taskid, epoch, model, optimizer, scheduler)

    if args.use_null_space and taskid + 1 < GVM.cl_mngr.num_tasks:
        _nstimer = ttime()
        print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: null-space update started", flush=True)
        new_interm_tensor_dict = get_interm_tensor_dict(GVM, model, GVM.cache_dict['null_param_id_dict'])
        GVM.cache_dict['interm_tensor_dict'] = accumulate_interm_tensor_dict(GVM, GVM.cache_dict['interm_tensor_dict'], new_interm_tensor_dict)
        del new_interm_tensor_dict
        GVM.cache_dict['update_proj_dict'] = get_update_projection_dict(GVM, GVM.cache_dict['null_param_id_dict'], GVM.cache_dict['interm_tensor_dict'])
        print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: null-space update took {misc.format_duration(ttime() - _nstimer)}", flush=True)

    if args.refine_head:
        extract_class_features(GVM, model)
        if args.refine_head:
            refine_head(GVM, model)

    print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")

    return model


def evaluate_one_task(GVM: GlobalVarsManager, train_taskid: int, eval_taskid: int, eval_task_classes: list[int], model: DefocusAttentionNetwork) -> OrderedDict[str, float]:
    use_amp: bool = GVM.args.use_amp
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type=GVM.args.transform_type, target_map_to_local=False)
    dataloader = DataLoader(dataset, batch_size=GVM.args.eval_batch_size, shuffle=False, num_workers=GVM.args.eval_workers, pin_memory=True, timeout=GVM.args.timeout if GVM.args.eval_workers > 0 else 0)

    set_model_mode(GVM, model, training=False)
    fwd = forward_module(GVM, model)
    scalar_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    for images, target in tqdm.tqdm(dataloader, total=len(dataloader), desc=f"Task [{train_taskid + 1}] eval task [{eval_taskid + 1}]", dynamic_ncols=True, disable=GVM.args.progress_interval <= 0, mininterval=GVM.args.progress_interval):
        images: Tensor = images.cuda(non_blocking=True)
        target: Tensor = target.cuda(non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                logits: Tensor = fwd(images)

        assert logits.ndim == 2
        assert logits.shape[1] == len(GVM.cl_mngr.sofar_task_classes), f"{logits.shape}, {len(GVM.cl_mngr.sofar_task_classes)}"

        class_inc_preds = logits.argmax(dim=1)

        acc_class_inc, acc_topnn_class, num_nn_class = misc.calc_acc_topnn_dynamically(class_inc_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_class_inc=acc_class_inc)

    assert len(dataset) == len(scalar_meter)
    result_dict = scalar_meter.update_epoch_average_value()

    print(f"Task [{train_taskid + 1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid + 1:>{len(_tt := str(train_taskid + 1))}}/{_tt}]: eval_time={ttime() - _ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")

    result_dict['num_samples'] = len(dataset)

    return result_dict


def evaluate_tasks_sofar(GVM: GlobalVarsManager, train_taskid: int, model: DefocusAttentionNetwork):
    model = modify_head(GVM, model, training=False)
    average_acc_meter = misc.ScalarMeter(acc_class_inc="samp_avg:>6.2%")

    for eval_taskid in range(GVM.cl_mngr.current_taskid + 1):
        eval_task_classes = GVM.cl_mngr.get_classes(eval_taskid)
        one_result_dict = evaluate_one_task(GVM, train_taskid, eval_taskid, eval_task_classes, model)
        GVM.acc_mat_dict[f'AccClassIncMat'][train_taskid, eval_taskid] = one_result_dict['acc_class_inc']
        average_acc_meter.add_step_value(**one_result_dict)

    avg_result_dict = average_acc_meter.update_epoch_average_value()
    GVM.acc_mat_dict[f'AccClassIncList'][train_taskid] = avg_result_dict['acc_class_inc']


def task_ending_info(GVM: GlobalVarsManager):
    current_taskid = GVM.cl_mngr.current_taskid

    acc_info_dict = {
        'class_inc_last_acc': float(GVM.acc_mat_dict['AccClassIncList'][current_taskid]),
        'class_inc_last_forg': misc.calc_forgetting(GVM.acc_mat_dict['AccClassIncMat'], current_taskid),
    }
    _formatter = misc.ScalarFormatter(sep=' | ', class_inc_last_acc=">6.2%", class_inc_last_forg=">6.2%")

    print(f":: ** Results of task [{current_taskid + 1}]: [ {_formatter(**acc_info_dict)} ] **")
    print(f":: ** Time so far: {misc.format_duration(ttime() - GVM.cache_dict['exp_start_time'])} **")

    if getattr(GVM.args, 'use_wandb', False):
        try:
            import wandb
            wandb.log({
                "task": current_taskid,
                "class_inc_last_acc": acc_info_dict['class_inc_last_acc'],
                "class_inc_last_forg": acc_info_dict['class_inc_last_forg']
            })
        except ImportError:
            pass



def get_param_id_dict(model: DefocusAttentionNetwork, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    param_id_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad and any([_s in n for _s in patterns]):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    assert len(param_id_dict) > 0, f"{param_id_dict}"
    return param_id_dict


def get_head_dim_arg_dict(GVM: GlobalVarsManager, args: argparse.Namespace) -> dict[Literal['num_classes'], int]:
    head_dim_arg_dict = {}
    head_dim_type = args.head_dim_type

    match args.logit_type:
        case 'sim_imgtext':
            assert head_dim_type in ('pretrained', 'text_dim')
        case 'head_out':
            assert head_dim_type in ('task_classes')

    match head_dim_type:
        case 'task_classes':
            head_dim_arg_dict['num_classes'] = len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.sofar_task_classes)
        case 'pretrained':
            pass
        case 'text_dim':
            head_dim_arg_dict['num_classes'] = 512
        case _:
            raise ValueError(head_dim_type)
    return head_dim_arg_dict


def modify_head(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, training: bool, **kwargs):
    args: argparse.Namespace = GVM.args

    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes

    if args.logit_type == 'head_out':
        if model.head.out_features != len(_target_classes):
            _mh = deepcopy(model.head)
            _mdevice = _mh.weight.device
            _mdtype = _mh.weight.dtype
            model.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
            model.head.requires_grad_(_mh.weight.requires_grad)

            if training:
                assert model.head.weight.requires_grad
            else:
                assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
                _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.head.weight.data.shape == _hw.shape
                model.head.weight.data = _hw

                if _mh.bias is not None:
                    _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                    assert model.head.bias.data.shape == _hb.shape
                    model.head.bias.data = _hb
        else:
            pass
    else:
        raise ValueError(args.logit_type)

    return model


def get_interm_tensor_dict(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, null_param_id_dict: dict) -> dict[int, Tensor]:
    interm_tensor_dict: dict[int, dict[str, Tensor]] = {}

    def _forward_hook(module: nn.Module, args: tuple[Tensor], output: Tensor):
        if isinstance(module, utils.mamba_block.IntermReader):
            _pid = module.dst_param_id
            if _pid in null_param_id_dict:
                _mname = module.module_name
                _interm_tensor: Tensor = args[0]  # [b*l, d]

                if _mname == 'interm_reader_step':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b l) d')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                elif _mname == 'interm_reader_B':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b l) d')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                elif _mname == 'interm_reader_C':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b l) d')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                elif _mname == 'interm_reader_A':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b d) l')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                elif _mname == 'interm_reader_out':
                    _interm_tensor = rearrange(_interm_tensor, 'b l d -> (b l) d')
                    _interm_tensor = torch.matmul(_interm_tensor.T, _interm_tensor) / _interm_tensor.shape[0]
                else:
                    raise ValueError()

                assert _mname.startswith('interm_reader_')

                if _pid not in interm_tensor_dict:
                    interm_tensor_dict[_pid] = {}
                if _mname not in interm_tensor_dict[_pid]:
                    interm_tensor_dict[_pid][_mname] = torch.zeros_like(_interm_tensor)
                interm_tensor_dict[_pid][_mname] += _interm_tensor
        else:
            raise NotImplementedError()

    _handle_list: list[torch.utils.hooks.RemovableHandle] = []
    for n, m in model.named_modules():
        if 'interm_reader' in n and isinstance(m, utils.mamba_block.IntermReader):
            _handle_list.append(m.register_forward_hook(_forward_hook))

    model = set_model_mode(GVM, model, training=False)

    args = GVM.args
    train_dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=args.seperate_head, use_eval_transform=True, expand_times=1)

    dataloader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    for nb, (img, _) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), desc="null-space: collect intermediates", dynamic_ncols=True, disable=args.progress_interval <= 0, mininterval=args.progress_interval):
        with torch.no_grad():
            img: Tensor
            model(img.cuda(non_blocking=True))

    for _h in _handle_list:
        _h.remove()

    assert len(interm_tensor_dict) > 0
    assert sorted(list(interm_tensor_dict.keys())) == sorted(list(null_param_id_dict.keys())), f"({len(interm_tensor_dict.keys())}){interm_tensor_dict.keys()}; ({len(null_param_id_dict)}){null_param_id_dict}"

    if (_k := 'interm_sample_list') not in GVM.cache_dict:
        GVM.cache_dict[_k] = []
    GVM.cache_dict[_k].append(len(dataloader.dataset))

    return interm_tensor_dict


def accumulate_interm_tensor_dict(GVM: GlobalVarsManager, cached_interm_tensor_dict: dict[int, dict[str, Tensor]], new_interm_tensor_dict: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    assert len(new_interm_tensor_dict) > 0
    args = GVM.args

    if cached_interm_tensor_dict == {}:
        merged_interm_tensor_dict = new_interm_tensor_dict
    else:
        assert (_lc := list(cached_interm_tensor_dict.keys())) == (_ln := list(new_interm_tensor_dict.keys())), f"{_lc}, {_ln}"
        merged_interm_tensor_dict: dict[int, Tensor] = {}
        for _pid in cached_interm_tensor_dict.keys():
            merged_interm_tensor_dict[_pid] = {}
            for _mname in cached_interm_tensor_dict[_pid].keys():
                _cached_tensor = cached_interm_tensor_dict[_pid][_mname]
                _new_tensor = new_interm_tensor_dict[_pid][_mname]
                assert _cached_tensor.shape == _new_tensor.shape

                match args.null_interm_accum:
                    case 'sum':
                        merged_interm_tensor_dict[_pid][_mname] = _cached_tensor + _new_tensor
                    case 'mean':
                        _num_list: list[int] = GVM.cache_dict['interm_sample_list']
                        merged_interm_tensor_dict[_pid][_mname] = sum(_num_list[:-1]) / sum(_num_list) * _cached_tensor + _num_list[-1] / sum(_num_list) * _new_tensor
                    case _:
                        raise ValueError()
    return merged_interm_tensor_dict


def get_update_projection_dict(GVM: GlobalVarsManager, null_param_id_dict: dict, interm_tensor_dict: dict[int, dict[str, Tensor]]) -> dict[int, dict[str, Tensor]]:
    args = GVM.args
    update_proj_dict = {}

    def adaptive_threshold(svals: torch.Tensor, offset: float = 0):
        points: np.ndarray = svals.cpu().numpy()
        assert points.ndim == 1
        if len(points) >= 128:
            fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
            _delta = 1
            diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            _drop_ratio = 0.03
            drop_num = int(len(points) * _drop_ratio / 2)
            assert len(points) - drop_num >= 10
            valid_o2 = diff_o2[drop_num:-drop_num]
            thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
        else:
            diff_o1 = points[:-1] - points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
        i_thres = np.arange(len(points))[points >= thres_val].max()
        if 0 <= offset < 1:
            i_thres = min(i_thres + int(offset * (len(points) - i_thres)), len(points) - 1)
        else:
            i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

        zero_idx = np.zeros(len(points), dtype=np.int64)
        zero_idx[i_thres:] = 1
        zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
        return zero_idx

    svals_dict = OrderedDict()
    zero_idx_dict = OrderedDict()
    for _pid in tqdm.tqdm(interm_tensor_dict.keys(), total=len(interm_tensor_dict), desc="null-space: SVD projections", dynamic_ncols=True, disable=GVM.args.progress_interval <= 0, mininterval=GVM.args.progress_interval):
        update_proj_dict[_pid] = {}
        for _mname in interm_tensor_dict[_pid].keys():
            U, S, Vt = torch.linalg.svd(interm_tensor_dict[_pid][_mname], full_matrices=True)  # A=U@diag(S)@Vt
            S: Tensor
            Vt: Tensor
            zero_idx = adaptive_threshold(S)
            zero_idx: torch.BoolTensor
            assert torch.count_nonzero(zero_idx) > 0, f"{zero_idx}, {type(zero_idx)}, {torch.count_nonzero(zero_idx)}"

            svals_dict[_dkey := f"{null_param_id_dict[_pid]['name']}-{_mname}"] = S.cpu().clone()
            zero_idx_dict[_dkey] = zero_idx.cpu().clone()

            basis = Vt[zero_idx]
            proj = basis.T @ basis
            proj = proj / torch.norm(proj)

            assert proj.shape[0] == proj.shape[1], proj.shape

            null_eta: float = args.null_eta
            update_proj_dict[_pid][_mname] = null_eta * proj.detach() + (1 - null_eta) * torch.eye(proj.shape[0], device=proj.device, dtype=proj.dtype)

    return update_proj_dict


def extract_class_features(GVM: GlobalVarsManager, model: DefocusAttentionNetwork) -> None:
    model = set_model_mode(GVM, model, training=False)

    dataset = define_dataset(GVM, GVM.cl_mngr.current_task_classes, training=True, transform_type=args.transform_type, target_map_to_local=False, use_eval_transform=True, expand_times=1)
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=args.timeout if args.eval_workers > 0 else 0)

    feats = torch.empty([len(dataset), model.embed_dim], dtype=torch.float32)
    label = torch.empty([len(dataset)], dtype=torch.long)

    smp_idx = 0
    for img, lbl in dataloader:
        with torch.no_grad():
            img: Tensor
            lbl: Tensor
            _feat = model.encode_image(img.cuda(non_blocking=True), pre_logits=True).cpu()
            for _f, _l in zip(_feat, lbl):
                feats[smp_idx] = _f
                label[smp_idx] = _l
                smp_idx += 1
    assert smp_idx == len(dataset)

    if GVM.args.refine_head:
        _mean_list = []
        _cov_list = []
        _class_list = []
        for _l in label.unique():
            _cls_feats = feats[label == _l]
            _mean_list.append(torch.mean(_cls_feats, dim=0, keepdim=False))
            _cov_list.append(torch.cov(torch.tensor(_cls_feats, dtype=torch.float64).T) + torch.eye(_cls_feats.shape[-1]) * 1e-4)
            _class_list.append(_l)
        _mean_list = torch.stack(_mean_list)
        _cov_list = torch.stack(_cov_list)
        _class_list = torch.stack(_class_list)

        _key = 'class_features'
        if _key not in GVM.cache_dict:
            GVM.cache_dict[_key] = {'mean': _mean_list, 'cov': _cov_list, 'class': _class_list}
        else:
            GVM.cache_dict[_key]['mean'] = torch.cat([GVM.cache_dict[_key]['mean'], _mean_list])
            GVM.cache_dict[_key]['cov'] = torch.cat([GVM.cache_dict[_key]['cov'], _cov_list])
            GVM.cache_dict[_key]['class'] = torch.cat([GVM.cache_dict[_key]['class'], _class_list])
            assert len(GVM.cache_dict[_key]['mean']) == len(GVM.cache_dict[_key]['cov']) == len(GVM.cache_dict[_key]['class']) == len(GVM.cl_mngr.sofar_task_classes)

    return None


def refine_head(GVM: GlobalVarsManager, model: DefocusAttentionNetwork):
    feats_mean: Tensor = GVM.cache_dict['class_features']['mean']
    feats_cov: Tensor = GVM.cache_dict['class_features']['cov']
    feats_class: Tensor = GVM.cache_dict['class_features']['class']
    assert len(feats_class.unique()) == len(GVM.cl_mngr.sofar_task_classes)

    stat_dataset = TensorDataset(feats_mean, feats_cov, feats_class)

    model = modify_head(GVM, model, training=False)
    mhead = model.head

    mhead.train()
    mhead.cuda()
    mhead.requires_grad_()

    optimizer = create_optimizer_v2(mhead, opt='sgd', lr=0.001, weight_decay=1e-4, momentum=0.9)
    scheduler, num_epochs = create_scheduler_v2(optimizer, 'multistep', num_epochs=GVM.args.refine_epochs, decay_milestones=[999,], decay_rate=0.1)
    criterion = nn.CrossEntropyLoss().cuda()
    from torch.distributions.multivariate_normal import MultivariateNormal

    scalar_meter = misc.ScalarMeter(loss="samp_avg:.4f", acc_top1="samp_avg:>6.2%")
    for epoch in range(1, num_epochs + 1):
        scheduler.step(epoch)

        smp_inp = []
        smp_tgt = []
        assert len(stat_dataset) == len(GVM.cl_mngr.sofar_task_classes)
        _ns = 256
        for _cmean, _ccov, _cclass in stat_dataset:
            if not GVM.args.rand_conv:
                m = MultivariateNormal(_cmean.float(), _ccov.float())
            else:
                _tmp = torch.randn([1, _cmean.shape[0]], device=_cmean.device)
                _tmp = torch.matmul(_tmp.T, _tmp)
                _tmp = torch.eye(_cmean.shape[0]) + _tmp
                m = MultivariateNormal(_cmean.float(), _tmp)
            _smp = m.sample(sample_shape=(_ns,))
            smp_inp.append(_smp)
            smp_tgt.append(torch.as_tensor([_cclass,] * _ns, dtype=torch.long))
        smp_inp = torch.cat(smp_inp)
        smp_tgt = torch.cat(smp_tgt)

        train_data = TensorDataset(smp_inp, smp_tgt)
        assert len(train_data) == len(stat_dataset) * _ns
        dataloader = DataLoader(train_data, batch_size=256, shuffle=True)

        for inp, tgt in dataloader:
            out: Tensor = mhead(inp.cuda(non_blocking=True))
            logits = out
            loss: Tensor = criterion(logits / GVM.args.temperature, tgt.cuda(non_blocking=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_top1, = misc.calc_accuracy(logits.cpu(), tgt.cpu(), topk=(1, ))
            scalar_meter.add_step_value(len(inp), loss=loss.item(), acc_top1=acc_top1)
        if (epoch % 10 == 0 or epoch == num_epochs):
            print(f":: {inspect.stack()[0].function}: epoch [{epoch}/{num_epochs}]: {scalar_meter.format_outout(scalar_meter.update_epoch_average_value())}")


CKPT_TASK, CKPT_EPOCH, RESULTS, WANDB_ID = 'task_end.pt', 'epoch.pt', 'results.json', 'wandb_id.txt'


def run_name(args: argparse.Namespace) -> str:
    return f"mamba_cl_{args.model}_{args.dataset}_t{args.num_tasks}_e{args.epochs}_seed_{args.seed}"


def build_vim_model(GVM: GlobalVarsManager) -> nn.Module:
    from utils import vim_mamba
    args = GVM.args
    path = args.pretrained_path
    if path == 'vim_small_in1k':
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(*vim_mamba.VIM_SMALL_IN1K)
    model = vim_mamba.MODELS[args.model](grad_ckpt=args.grad_ckpt)

    missing, unexpected = vim_mamba.load_finetune(model, path)
    print(f":: loaded {args.model} from {path}: missing keys {missing}, unexpected keys {unexpected}")
    if bad := [k for k in missing if not k.startswith('head.')]:
        raise RuntimeError(f"pretrained weights did not load: {len(bad)} backbone keys missing, e.g. {bad[:4]}")
    setattr(model, 'pretrained_cfg', {'pretrained_path': args.pretrained_path})
    GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)
    model.reload_params()

    # Vim's backward-direction A is A_b_log, which the default 'A_log' pattern does not match
    for _k in ('training_string', 'null_patterns'):
        _pats = tuple(getattr(args, _k))
        if 'A_log' in _pats and 'A_b_log' not in _pats:
            setattr(args, _k, _pats + ('A_b_log',))
            print(f":: {args.model}: --{_k} {' '.join(getattr(args, _k))}")
    GVM.cache_dict['training_string'] = args.training_string
    return model


def build_pretrained_model(GVM: GlobalVarsManager) -> DefocusAttentionNetwork:
    args = GVM.args
    if args.model.startswith('vim_'):
        return build_vim_model(GVM)
    cfg_name = "defocus_mamba_large_22k.yaml" if args.model == 'defocus_mamba_large' else f"{args.model}.yaml"
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", cfg_name)
    model: DefocusAttentionNetwork = build_model(get_config(config_path))

    ckpt = torch.load(args.pretrained_path, map_location='cpu')
    load_info = model.load_state_dict(ckpt['model'], strict=False)
    print(f":: loaded {args.model}: missing keys {load_info.missing_keys}, unexpected keys {load_info.unexpected_keys}")
    setattr(model, 'pretrained_cfg', {})
    model.pretrained_cfg['pretrained_path'] = args.pretrained_path
    GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)
    model.reload_params()
    GVM.cache_dict['training_string'] = args.training_string
    return model


def rng_state() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def set_rng_state(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'][:torch.cuda.device_count()])


def atomic_save(obj: Any, path: str) -> None:
    # write-then-rename, so a session killed mid-save never leaves a truncated checkpoint
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def load_model_state(model: DefocusAttentionNetwork, state: dict[str, Tensor], strict: bool = True) -> None:
    # the head is rebuilt per task, so match its saved shape before loading
    _w = state['head.weight']
    if model.head.weight.shape != _w.shape:
        _mh = model.head
        model.head = _mh.__class__(_w.shape[1], _w.shape[0], 'head.bias' in state, _mh.weight.device, _mh.weight.dtype)
    load_info = model.load_state_dict(state, strict=strict)
    assert not load_info.unexpected_keys, load_info.unexpected_keys


def save_epoch_checkpoint(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, optimizer: torch.optim.Optimizer, taskid: int, epoch: int) -> None:
    if GVM.run_dir is None:
        return
    # only trained params change within a task; the rest comes from task_end.pt or the pretrained weights
    atomic_save({'task': taskid, 'epoch': epoch,
                 'model': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                 'optimizer': optimizer.state_dict(), 'rng': rng_state()}, os.path.join(GVM.run_dir, CKPT_EPOCH))
    if GVM.hub is not None and GVM.args.hf_sync_epochs:
        GVM.hub.push(CKPT_EPOCH, f"task {taskid + 1} epoch {epoch}")


def write_results(GVM: GlobalVarsManager, taskid: int) -> None:
    # accuracies are fractions in [0, 1]; forgetting at task t only looks at rows 0..t
    _mat = GVM.acc_mat_dict['AccClassIncMat']
    _nt = taskid + 1
    results = {
        'run_name': run_name(GVM.args),
        'tasks_done': _nt,
        'num_tasks': GVM.cl_mngr.num_tasks,
        'finished': _nt == GVM.cl_mngr.num_tasks,
        'avg_acc': float(GVM.acc_mat_dict['AccClassIncList'][taskid]),
        'forgetting': float(misc.calc_forgetting(_mat[:_nt], taskid)),
        'avg_acc_per_task': [float(_a) for _a in GVM.acc_mat_dict['AccClassIncList'][:_nt]],
        'forgetting_per_task': [float(misc.calc_forgetting(_mat[:_t + 1], _t)) for _t in range(_nt)],
        'acc_matrix': _mat[:_nt, :_nt].tolist(),
        'elapsed_hours': (ttime() - GVM.cache_dict['exp_start_time']) / 3600,
        'args': vars(GVM.args),
    }
    with open(os.path.join(GVM.run_dir, RESULTS), 'w') as f:
        json.dump(results, f, indent=2, default=str)


def save_task_checkpoint(GVM: GlobalVarsManager, model: DefocusAttentionNetwork, taskid: int) -> None:
    if GVM.run_dir is None:
        return
    _skip = ('update_proj_dict', 'null_param_id_dict', 'interm_tensor_dict', 'exp_start_time', 'resume_epoch_state')
    null_space = None
    if GVM.args.use_null_space:
        # these dicts are keyed by id(param), which changes between processes, so store parameter names instead
        _id2name = {id(p): n for n, p in model.named_parameters()}
        null_space = {'param_names': [_v['name'] for _v in GVM.cache_dict['null_param_id_dict'].values()],
                      'interm': {_id2name[_pid]: {_k: _t.cpu() for _k, _t in _d.items()} for _pid, _d in GVM.cache_dict['interm_tensor_dict'].items()}}
    atomic_save({'tasks_done': taskid + 1,
                 'model': model.state_dict(),
                 'param_dict': GVM.param_dict,
                 'acc_mat_dict': GVM.acc_mat_dict,
                 'cache_dict': {_k: _v for _k, _v in GVM.cache_dict.items() if _k not in _skip},
                 'null_space': null_space,
                 'elapsed': ttime() - GVM.cache_dict['exp_start_time'],
                 'rng': rng_state()}, os.path.join(GVM.run_dir, CKPT_TASK))
    write_results(GVM, taskid)
    if os.path.exists(_p := os.path.join(GVM.run_dir, CKPT_EPOCH)):
        os.remove(_p)
    print(f":: checkpoint saved: {GVM.run_dir} (tasks done: {taskid + 1})", flush=True)

    if GVM.hub is not None:
        GVM.hub.push(CKPT_TASK, f"task {taskid + 1} done")
        GVM.hub.push(RESULTS, f"results after task {taskid + 1}")
        GVM.hub.squash_history()


def setup_checkpointing(GVM: GlobalVarsManager, model: DefocusAttentionNetwork) -> int:
    """Pull this run's files from the HF mirror if they are not on disk, then restore them. Returns the number of finished tasks."""
    args = GVM.args
    if not args.ckpt_dir:
        assert not args.hf_repo, "--hf_repo needs --ckpt_dir"
        return 0
    GVM.run_dir = os.path.join(args.ckpt_dir, run_name(args))
    os.makedirs(GVM.run_dir, exist_ok=True)
    if args.hf_repo:
        from utils.hf_sync import HubMirror
        GVM.hub = HubMirror(args.hf_repo, run_name(args), args.ckpt_dir, squash=args.hf_squash)
        if _pulled := GVM.hub.pull([CKPT_TASK, CKPT_EPOCH, RESULTS, WANDB_ID]):
            print(f":: pulled {_pulled} from hf://{args.hf_repo}/{run_name(args)}")

    tasks_done = 0
    if os.path.exists(_p := os.path.join(GVM.run_dir, CKPT_TASK)):
        ckpt = torch.load(_p, map_location='cpu', weights_only=False)
        tasks_done = ckpt['tasks_done']
        load_model_state(model, ckpt['model'])
        GVM.param_dict, GVM.acc_mat_dict = ckpt['param_dict'], ckpt['acc_mat_dict']
        GVM.cache_dict.update(ckpt['cache_dict'])
        GVM.cache_dict['exp_start_time'] = ttime() - ckpt['elapsed']
        set_rng_state(ckpt['rng'])
        if args.use_null_space and tasks_done < GVM.cl_mngr.num_tasks:
            _params = dict(model.named_parameters())
            GVM.cache_dict['null_param_id_dict'] = {id(_params[n]): {'name': n, 'shape': list(_params[n].shape)} for n in ckpt['null_space']['param_names']}
            GVM.cache_dict['interm_tensor_dict'] = {id(_params[n]): {_k: _t.cuda() for _k, _t in _d.items()} for n, _d in ckpt['null_space']['interm'].items()}
            # projections are a deterministic function of the accumulated intermediates, so recompute rather than store them
            GVM.cache_dict['update_proj_dict'] = get_update_projection_dict(GVM, GVM.cache_dict['null_param_id_dict'], GVM.cache_dict['interm_tensor_dict'])
        GVM.cache_dict.setdefault('update_proj_dict', {})
        print(f":: resumed from {_p}: {tasks_done}/{GVM.cl_mngr.num_tasks} tasks done", flush=True)

    if os.path.exists(_p := os.path.join(GVM.run_dir, CKPT_EPOCH)):
        _epoch_ckpt = torch.load(_p, map_location='cpu', weights_only=False)
        if _epoch_ckpt['task'] == tasks_done:  # older ones belong to a task that already finished
            GVM.cache_dict['resume_epoch_state'] = _epoch_ckpt
    return tasks_done


def init_wandb(GVM: GlobalVarsManager) -> None:
    import wandb
    _id_path = os.path.join(GVM.run_dir, WANDB_ID) if GVM.run_dir else None
    if _id_path and os.path.exists(_id_path):
        with open(_id_path) as f:
            run_id = f.read().strip()
    else:
        run_id = uuid.uuid4().hex[:12]  # wandb.util.generate_id is gone in newer wandb releases
    wandb.init(project="mamba-cl-baseline", name=run_name(GVM.args), config=vars(GVM.args), id=run_id, resume='allow')
    if _id_path and not os.path.exists(_id_path):
        with open(_id_path, 'w') as f:
            f.write(run_id)
        if GVM.hub is not None:
            GVM.hub.push(WANDB_ID, "wandb run id")


if __name__ == "__main__":
    args = get_args()
    seed_etc_options(args.seed)

    GVM = GlobalVarsManager()
    GVM.init_from_args(args)
    GVM.cache_dict['exp_start_time'] = ttime()

    model = build_pretrained_model(GVM)
    if args.data_parallel:
        print(f":: data_parallel: train/eval batches split across {torch.cuda.device_count()} GPU(s)")
    tasks_done = setup_checkpointing(GVM, model)
    if tasks_done == GVM.cl_mngr.num_tasks:
        print(f":: all {tasks_done} tasks already done; results in {GVM.run_dir}/{RESULTS}")

    if getattr(args, 'use_wandb', False):
        try:
            init_wandb(GVM)
        except ImportError:
            print("wandb package not found. Skipping wandb initialization.")

    for taskid, current_task_classes in GVM.cl_mngr:
        GVM.update_label_maps(taskid, current_task_classes)
        if taskid < tasks_done:
            continue
        model = train_one_task(GVM, taskid, current_task_classes, model)

        evaluate_tasks_sofar(GVM, taskid, model)
        task_ending_info(GVM)
        save_task_checkpoint(GVM, model, taskid)

    if getattr(args, 'use_wandb', False):
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass
