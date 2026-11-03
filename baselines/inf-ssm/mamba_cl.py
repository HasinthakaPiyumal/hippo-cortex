"""Mamba-CL (Cheng et al., IJCV 2025) as a --cl_method of the Inf-SSM trainer, so it runs under exactly
the Inf-SSM protocol (shared 100-way head, AdamW + cosine, mixup/cutmix, best-val epoch, same splits).

What is kept from Mamba-CL (baselines/mamba-cl/train_eval.py, utils/mod_adam.py):
- the model is baselines/mamba-cl/utils/vim_mamba.py (same weights and computation as mamba_vim.py, with
  x_proj split into _step/_B/_C and IntermReaders that expose the input of each projected parameter);
- only the head and the null-space parameters are trained: x_proj_{step,B,C}, A_log / A_b_log and
  out_proj.weight of every block (Mamba-CL's --training_string); the rest of the backbone is frozen;
- after each task the inputs of those parameters are collected over the task's train images (eval
  transform, every parameter frozen), summed into one uncentred covariance per parameter over all tasks
  so far, and their null space is found with Mamba-CL's adaptive SVD threshold;
- every optimizer update of a projected parameter is multiplied by
  P = eta * (basis^T basis / ||basis^T basis||) + (1 - eta) * I: on the right for x_proj_* and out_proj
  (input dimension), on the left for A_log (rows). ModAdam projects the Adam update; here the whole
  AdamW step (decoupled weight decay included) is projected, which is the same for any optimizer.

What comes from the Inf-SSM protocol instead of Mamba-CL's: the head, optimizer, LR schedule, epochs,
augmentation, image size, batch size, best-epoch selection and evaluation.
"""
import copy
import importlib
import importlib.util
import os
import sys

import numpy as np
import torch
import tqdm
from einops import rearrange
from torch import nn, Tensor
from torch.utils.data import DataLoader, SequentialSampler

MCL_UTILS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mamba-cl/utils"))
TRAIN_PATTERNS = ("head", "x_proj", "out_proj.weight", "A_log", "A_b_log")  # Mamba-CL's --training_string for Vim
# reader name -> which side of the update the projection multiplies (ModAdam: A on the left, the rest on the right)
READER_SIDE = {"interm_reader_step": "right", "interm_reader_B": "right", "interm_reader_C": "right",
               "interm_reader_out": "right", "interm_reader_A": "left"}


def load_vim_module():
    """baselines/mamba-cl/utils/vim_mamba.py, loaded under a private package name (every baseline has a 'utils')."""
    name = "_mamba_cl_utils"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(MCL_UTILS, "__init__.py"), submodule_search_locations=[MCL_UTILS])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{name}.vim_mamba"), importlib.import_module(f"{name}.interm_reader")


def build_model(model_name: str, ckpt_path: str | None, **kwargs) -> nn.Module:
    """Mamba-CL's Vim, checkpoint loaded, x_proj split, everything but Mamba-CL's parameters frozen."""
    vim, _ = load_vim_module()
    model = vim.MODELS[model_name](**kwargs)
    if ckpt_path is not None:
        missing, unexpected = vim.load_finetune(model, ckpt_path)
        print(f":: loaded {ckpt_path}: missing={missing}, unexpected={unexpected}")
    model.reload_params()
    for n, p in model.named_parameters():
        p.requires_grad_(any(s in n for s in TRAIN_PATTERNS))
    return model


def adaptive_threshold(svals: Tensor) -> Tensor:
    """Mamba-CL's get_update_projection_dict.adaptive_threshold (offset 0): True for the singular
    directions that form the null space (from the knee of the spectrum on)."""
    import scipy.ndimage  # here, so that the reg / seq runs do not need scipy
    points: np.ndarray = svals.cpu().numpy()
    if len(points) >= 128:
        fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
        diff_o1 = fil_points[:-1] - fil_points[1:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        drop_num = int(len(points) * 0.03 / 2)
        assert len(points) - drop_num >= 10
        valid_o2 = diff_o2[drop_num:-drop_num]
        thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
    else:
        diff_o1 = points[:-1] - points[1:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
    i_thres = np.arange(len(points))[points >= thres_val].max()
    zero_idx = np.zeros(len(points), dtype=bool)
    zero_idx[i_thres:] = True
    return torch.from_numpy(zero_idx).to(svals.device)


def batch_cov(reader_name: str, x: Tensor) -> Tensor:
    """Mamba-CL's per-batch statistic of one reader input (get_interm_tensor_dict)."""
    x = x.float()
    if reader_name == "interm_reader_A":  # delta (b, d_inner, l): rows are (b, l), columns d_inner
        x = rearrange(x, "b l d -> (b d) l")
    else:
        x = rearrange(x, "b l d -> (b l) d")
    return x.T @ x / x.shape[0]


class NullSpace:
    """Covariances of the projected parameters' inputs over all tasks so far, and the update projections."""

    def __init__(self, model: nn.Module, eta: float):
        _, interm_reader = load_vim_module()
        self.eta = eta
        names = {id(p): n for n, p in model.named_parameters()}
        self.readers = {}  # module name -> (parameter name, reader name)
        for n, m in model.named_modules():
            if isinstance(m, interm_reader.IntermReader):
                self.readers[n] = (names[m.dst_param_id], m.module_name)
        assert self.readers, "no IntermReaders: build the model with mamba_cl.build_model"
        self.cov: dict[str, Tensor] = {}   # parameter name -> summed covariance
        self.proj: dict[str, Tensor] = {}  # parameter name -> P
        self.side = {pn: READER_SIDE[rn] for pn, rn in self.readers.values()}
        self.svals: dict[str, Tensor] = {}
        self.null_dims: dict[str, int] = {}

    @torch.no_grad()
    def collect(self, model: nn.Module, dataset, batch_size: int, workers: int, device, progress_interval: float = 0) -> None:
        """Add the current task's covariances (dataset: the task's train images with the eval transform)."""
        new: dict[str, Tensor] = {}
        modules = dict(model.named_modules())

        def hook(module_name):
            pname, rname = self.readers[module_name]

            def _hook(module, args, output):
                c = batch_cov(rname, args[0])
                new[pname] = new[pname] + c if pname in new else c
            return _hook

        handles = [modules[n].register_forward_hook(hook(n)) for n in self.readers]
        was_training = model.training
        grad_state = {n: p.requires_grad for n, p in model.named_parameters()}
        model.eval()
        for p in model.parameters():  # vim_mamba feeds the A and B readers only with every parameter frozen
            p.requires_grad_(False)
        try:
            loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, num_workers=workers, pin_memory=True)
            for images, _ in tqdm.tqdm(loader, desc="null-space: collect intermediates", dynamic_ncols=True,
                                       disable=progress_interval <= 0, mininterval=progress_interval):
                model(images.to(device, non_blocking=True))
        finally:
            for h in handles:
                h.remove()
            for n, p in model.named_parameters():
                p.requires_grad_(grad_state[n])
            model.train(was_training)
        assert set(new) == set(self.side), f"readers that never fired: {set(self.side) - set(new)}"
        for pname, c in new.items():  # Mamba-CL's default --null_interm_accum sum
            self.cov[pname] = self.cov[pname] + c if pname in self.cov else c
        self.update_projections()

    @torch.no_grad()
    def update_projections(self) -> None:
        for pname, c in self.cov.items():
            _, S, Vt = torch.linalg.svd(c, full_matrices=True)
            zero_idx = adaptive_threshold(S)
            basis = Vt[zero_idx]
            proj = basis.T @ basis
            proj = proj / torch.norm(proj)
            self.proj[pname] = self.eta * proj + (1 - self.eta) * torch.eye(proj.shape[0], device=proj.device, dtype=proj.dtype)
            self.svals[pname] = S.cpu()
            self.null_dims[pname] = int(zero_idx.sum())

    def params(self, model: nn.Module) -> list[tuple[nn.Parameter, Tensor, str]]:
        """(parameter, projection, side) for every projected parameter that is being trained."""
        named = dict(model.named_parameters())
        return [(named[n], P, self.side[n]) for n, P in self.proj.items() if named[n].requires_grad]

    @staticmethod
    def snapshot(plist) -> list[Tensor]:
        return [p.detach().clone() for p, _, _ in plist]

    @staticmethod
    @torch.no_grad()
    def project_(plist, before: list[Tensor]) -> None:
        """Replace the update each parameter just received with its projection."""
        for (p, P, side), p0 in zip(plist, before):
            delta = p - p0
            delta = P.to(delta.dtype) @ delta if side == "left" else delta @ P.to(delta.dtype)
            p.copy_(p0 + delta)

    def state_dict(self) -> dict:
        return {'eta': self.eta, 'cov': {n: c.cpu() for n, c in self.cov.items()}}

    def load_state_dict(self, state: dict, device) -> None:
        assert state['eta'] == self.eta, f"checkpoint eta {state['eta']} != --null_eta {self.eta}"
        assert set(state['cov']) == set(self.side), "checkpoint covariances do not match the model's projected parameters"
        self.cov = {n: c.to(device) for n, c in state['cov'].items()}
        self.update_projections()


def eval_view(train_dataset, eval_transform):
    """The same train split with the eval transform (Mamba-CL collects with use_eval_transform=True)."""
    ds = copy.copy(train_dataset)
    ds.transform = eval_transform
    return ds
