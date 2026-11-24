"""
MambaBackbone — Vim-small (default) or De-focus Mamba base encoder + classification head(s).

OWNER: Praveen Dedigama

arch="vim_small" (default) is the network of both baselines (Inf-SSM and Mamba-CL under the
Inf-SSM protocol): baselines/mamba-cl/utils/vim_mamba.py, the same weights and computation as
baselines/inf-ssm/mamba_vim.py with x_proj split for the null-space parameters. Weights come
from the authors' task-0 checkpoint (``checkpoint_path="hf"``: HF repo ileening0022/Inf-SSM,
<hf_dir>/task_0_checkpoint_seed_<seed>.pth), so HippoCortex starts where the baselines start.
arch="defocus_base" is the older De-focus Mamba base network of Mamba-CL's own protocol.

Heads (``head_mode``):
- "shared":   one Linear over all ``n_classes_total`` classes, as in the Inf-SSM protocol. With
              the task-0 checkpoint the head comes from the checkpoint (it already knows task 0).
              forward(x, task_id) returns the logits of every class seen up to ``task_id``;
              labels must be shifted by label_offset(task_id) to global indices.
- "per_task": one head per task, earlier heads frozen (Mamba-CL --seperate_head). forward()
              returns the head of ``task_id`` only; label_offset() is 0.

Interface (used by the trainer):
- forward(x, task_id=None)  -> (logits, hidden). x: (B, 3, H, W), resized to ``image_size``.
                               hidden: model.encode_image(x, pre_logits=True), (B, feature_dim):
                               384 for Vim-small, 768 for De-focus base.
- extract_hidden(x)         -> hidden only.
- set_task_head(n_classes)  -> registers task ``task_id`` and returns the head(s) to train.
- label_offset(task_id)     -> add to the loader's local labels for forward()'s logits.
- task_logits(hidden, t)    -> task t's own classes only (task-incremental eval).
- joined_logits(hidden)     -> every class seen so far, in task order; with class_offset(t)
                               this is the class-incremental evaluation.

Training scope matches Mamba-CL's --training_string: x_proj*, out_proj.weight, A_log and, for
Vim's backward direction, A_b_log (plus the head). Everything else is frozen.

Loading order mirrors the baselines: build -> load checkpoint (strict=False) -> report keys ->
reload_params(). reload_params() splits ``x_proj`` into x_proj_step / x_proj_B / x_proj_C, which
the forward pass uses, so it must run after the checkpoint (which stores ``x_proj``) is loaded.

Vim-small runs on CPU through pure-PyTorch fallbacks (slow, fine for tiny test models). De-focus
base needs mamba-ssm 1.2.2 + causal-conv1d (CUDA); for CPU tests pass a stand-in via
``MambaBackbone(model=...)`` (see hippocortex/models/fake_defocus.py).
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAMBA_CL_UTILS = _REPO_ROOT / "baselines" / "mamba-cl" / "utils"
_PKG = "_mamba_cl_utils"  # private import name: baselines/*/utils would otherwise all be "utils"

DEFAULT_CONFIG_PATH = _MAMBA_CL_UTILS / "defocus_mamba_base.yaml"
CHECKPOINT_URL = (
    "https://github.com/OpenGVLab/De-focus-Attention-Networks/releases/download/v1.0/"
    "defocus_mamba_base_cls.pth"
)

TASK0_HF_REPO = "ileening0022/Inf-SSM"  # <hf_dir>/task_0_checkpoint_seed_<seed>.pth (seed 0 only)

ARCHS = ("vim_small", "defocus_base")
HEAD_MODES = ("shared", "per_task")

# Mamba-CL's --training_string minus 'head' (heads are handled separately). A_b_log is Vim's
# backward-direction A, which the plain 'A_log' pattern does not match.
BACKBONE_TRAINABLE_PATTERNS = ("x_proj", "out_proj.weight", "A_log", "A_b_log")

# If any of these are missing after loading, the pretrained weights did not load.
REQUIRED_KEY_PREFIXES = {
    "defocus_base": ("blocks.", "patch_embed.", "norm.", "cls_token"),
    "vim_small": ("layers.", "patch_embed.", "norm_f.", "cls_token", "pos_embed"),
}


def _load_mamba_cl_module(name: str) -> ModuleType:
    """Import baselines/mamba-cl/utils as a private package and return its ``name`` module."""
    full_name = f"{_PKG}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    if _PKG not in sys.modules:
        init = _MAMBA_CL_UTILS / "__init__.py"
        if not init.is_file():
            raise FileNotFoundError(
                f"Mamba-CL model code not found at {_MAMBA_CL_UTILS}. "
                "MambaBackbone imports it from the repository checkout (baselines/mamba-cl)."
            )
        spec = importlib.util.spec_from_file_location(
            _PKG, init, submodule_search_locations=[str(_MAMBA_CL_UTILS)]
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_PKG] = pkg
        try:
            spec.loader.exec_module(pkg)
        except BaseException:
            del sys.modules[_PKG]
            raise
    try:
        return importlib.import_module(full_name)
    except BaseException:
        sys.modules.pop(full_name, None)
        raise


def _load_mamba_cl_config_module() -> ModuleType:
    return _load_mamba_cl_module("config")


def check_load_report(
    missing: list[str], unexpected: list[str], arch: str = "defocus_base"
) -> None:
    """Print the load report and raise if the backbone weights did not load."""
    print(f":: loaded {arch}: missing keys {missing}, unexpected keys {unexpected}")
    bad = [k for k in missing if k.startswith(REQUIRED_KEY_PREFIXES[arch])]
    if bad:
        raise RuntimeError(
            f"Pretrained weights did not load: {len(bad)} required keys are missing "
            f"(e.g. {bad[:4]}). Check the checkpoint file and the model config."
        )


def task0_checkpoint(hf_dir: str = "cifar100_10task", seed: int = 0) -> str:
    """
    Download (or reuse the HF cache of) the authors' task-0 Vim-small checkpoint, the one
    both baselines start from. Only seed 0 was released.
    """
    from huggingface_hub import hf_hub_download

    filename = f"{hf_dir}/task_0_checkpoint_seed_{seed}.pth"
    try:
        return hf_hub_download(TASK0_HF_REPO, filename)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {TASK0_HF_REPO}/{filename} ({type(exc).__name__}). "
            "The authors released seed 0 only."
        ) from exc


def _read_checkpoint(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nDownload it with:\n  wget -O {path} {CHECKPOINT_URL}"
        )
    # weights_only: the checkpoint is a plain tensor dict; do not unpickle arbitrary objects.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


def _build_defocus_model(config_path: Path, checkpoint_path: Path) -> nn.Module:
    cfg_module = _load_mamba_cl_config_module()
    try:
        model = cfg_module.build_model(cfg_module.get_config(str(config_path)))
    except AssertionError as exc:  # MambaMixer asserts the CUDA fast path exists
        raise RuntimeError(
            "De-focus Mamba base needs mamba-ssm==1.2.2 and causal-conv1d (CUDA) to be "
            "constructed. On CPU pass a stand-in: MambaBackbone(model=TinyDefocus())."
        ) from exc

    info = model.load_state_dict(_read_checkpoint(checkpoint_path), strict=False)
    check_load_report(list(info.missing_keys), list(info.unexpected_keys), "defocus_base")
    model.reload_params()
    return model


def _build_vim_small(
    checkpoint_path: Path | None,
    num_classes: int,
    img_size: int = 224,
    grad_ckpt: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Mamba-CL's Vim-small with the checkpoint loaded the way the baselines load it
    (vim_mamba.load_finetune: a head of another shape is dropped), then x_proj split.
    ``kwargs`` (e.g. depth, embed_dim) are for small test models only.
    """
    vim = _load_mamba_cl_module("vim_mamba")
    model = vim.VisionMamba(
        img_size=img_size,
        patch_size=16,
        embed_dim=kwargs.pop("embed_dim", 384),
        depth=kwargs.pop("depth", 24),
        num_classes=num_classes,
        drop_path_rate=kwargs.pop("drop_path_rate", 0.0),  # Inf-SSM protocol: --drop_path 0
        grad_ckpt=grad_ckpt,
        **kwargs,
    )
    if checkpoint_path is not None:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        missing, unexpected = vim.load_finetune(model, str(checkpoint_path))
        check_load_report(missing, unexpected, "vim_small")
    model.reload_params()
    return model


class MambaBackbone(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        model: nn.Module | None = None,
        image_size: int = 224,
        arch: str = "vim_small",
        head_mode: str = "per_task",
        n_classes_total: int | None = None,
        grad_ckpt: bool = False,
        hf_dir: str = "cifar100_10task",
        seed: int = 0,
    ) -> None:
        """
        Args:
            checkpoint_path: Weights to load. "hf" = the authors' task-0 Vim-small checkpoint
                             (``hf_dir``, ``seed``); a path = that file. Required for
                             arch="defocus_base" unless ``model`` is given; None for Vim-small
                             = random init (tests only).
            config_path:     De-focus yaml (arch="defocus_base" only); defaults to
                             baselines/mamba-cl/utils/defocus_mamba_base.yaml.
            model:           Pre-built network exposing ``embed_dim``, ``head`` and
                             ``encode_image(x, pre_logits=True)``. Skips building and loading.
            image_size:      Inputs are bicubic-resized to image_size x image_size.
            arch:            "vim_small" or "defocus_base".
            head_mode:       "shared" (one head over n_classes_total classes) or "per_task".
            n_classes_total: Classes over all tasks; required for head_mode="shared".
            grad_ckpt:       Vim-small: recompute each block in backward (less GPU memory).
            hf_dir, seed:    Which task-0 checkpoint ``checkpoint_path="hf"`` downloads.
        """
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
        if head_mode not in HEAD_MODES:
            raise ValueError(f"head_mode must be one of {HEAD_MODES}, got {head_mode!r}")
        if head_mode == "shared" and not n_classes_total:
            raise ValueError('head_mode="shared" needs n_classes_total.')

        if model is None:
            if arch == "defocus_base":
                if checkpoint_path is None:
                    raise ValueError("Pass checkpoint_path (real model) or model (stand-in).")
                model = _build_defocus_model(
                    Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH,
                    Path(checkpoint_path),
                )
            else:
                if checkpoint_path == "hf":
                    checkpoint_path = task0_checkpoint(hf_dir, seed)
                # A head of n_classes_total classes keeps the checkpoint's task-0 head.
                model = _build_vim_small(
                    Path(checkpoint_path) if checkpoint_path is not None else None,
                    num_classes=n_classes_total or 1000,
                    img_size=image_size,
                    grad_ckpt=grad_ckpt,
                )

        self.arch = arch
        self.head_mode = head_mode
        self.image_size = image_size
        self.feature_dim = int(model.embed_dim)
        self.n_classes_total = n_classes_total

        # Classes of every task registered so far (both modes), in task order.
        self._task_classes: list[int] = []
        self.heads = nn.ModuleDict()  # per_task: one head per task
        self.shared_head: nn.Linear | None = None
        if head_mode == "shared":
            old = getattr(model, "head", None)
            if (
                isinstance(old, nn.Linear)
                and old.in_features == self.feature_dim
                and old.out_features == n_classes_total
            ):
                self.shared_head = old  # e.g. the task-0 checkpoint's head
            else:
                ref = next(model.parameters())
                self.shared_head = nn.Linear(self.feature_dim, n_classes_total).to(
                    device=ref.device, dtype=ref.dtype
                )
        # The model's own head is replaced by the head(s) above.
        model.head = nn.Identity()
        self.model = model

        for name, p in self.model.named_parameters():
            p.requires_grad_(any(pat in name for pat in BACKBONE_TRAINABLE_PATTERNS))

    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------

    @property
    def n_tasks(self) -> int:
        """Number of tasks registered with set_task_head() so far."""
        return len(self._task_classes)

    def set_task_head(self, n_classes: int, task_id: int | None = None) -> nn.Linear:
        """
        Register task ``task_id`` with ``n_classes`` classes and return the head to train.

        per_task: adds a new head for the task; earlier heads are kept and frozen.
        shared:   returns the shared head (trainable); the task's classes are the next
                  ``n_classes`` columns of it.

        Args:
            n_classes: Classes of the task.
            task_id:   Task index. None appends as the next task. An existing index
                       re-registers that task (and drops later ones); a gap is an error.

        Returns:
            The head to train. A new per-task head is NOT in any previously built
            optimizer; the caller must build or extend the optimizer with it.
        """
        if task_id is None:
            task_id = len(self._task_classes)
        if not 0 <= task_id <= len(self._task_classes):
            raise ValueError(
                f"task_id={task_id} leaves a gap; {len(self._task_classes)} tasks exist."
            )

        classes = self._task_classes[:task_id] + [int(n_classes)]
        if self.head_mode == "shared":
            if sum(classes) > self.n_classes_total:
                raise ValueError(
                    f"Tasks 0..{task_id} have {sum(classes)} classes but the shared head has "
                    f"{self.n_classes_total}."
                )
            self._task_classes = classes
            self.shared_head.requires_grad_(True)
            return self.shared_head

        for key in [k for k in self.heads.keys() if int(k) > task_id]:
            del self.heads[key]
        ref = next(self.model.parameters())
        head = nn.Linear(self.feature_dim, n_classes).to(device=ref.device, dtype=ref.dtype)
        self.heads[str(task_id)] = head
        self._task_classes = classes

        for key, h in self.heads.items():
            h.requires_grad_(key == str(task_id))
            if key != str(task_id):
                # No optimizer owns a frozen head, so nothing would ever clear its old grads.
                for p in h.parameters():
                    p.grad = None
        return head

    def _check_task(self, task_id: int | None) -> int:
        if not self._task_classes:
            raise RuntimeError("No task head yet; call set_task_head() first.")
        t = len(self._task_classes) - 1 if task_id is None else int(task_id)
        if not 0 <= t < len(self._task_classes):
            raise KeyError(f"No head for task {task_id}; {len(self._task_classes)} tasks exist.")
        return t

    def _head(self, task_id: int | None) -> nn.Linear:
        t = self._check_task(task_id)
        return self.shared_head if self.head_mode == "shared" else self.heads[str(t)]

    def head_parameters(self, task_id: int | None = None) -> list[nn.Parameter]:
        return list(self._head(task_id).parameters())

    def class_offset(self, task_id: int) -> int:
        """
        Index of task ``task_id``'s first class in joined_logits(): the number of classes
        of all earlier tasks. Global label = class_offset(task_id) + local label.
        """
        t = self._check_task(task_id)
        return sum(self._task_classes[:t])

    def label_offset(self, task_id: int | None = None) -> int:
        """Shift from the loader's local labels to the label space of forward()'s logits."""
        t = self._check_task(task_id)
        return self.class_offset(t) if self.head_mode == "shared" else 0

    def task_logits(self, hidden: Tensor, task_id: int | None = None) -> Tensor:
        """Logits of task ``task_id``'s own classes only (task-incremental evaluation)."""
        t = self._check_task(task_id)
        if self.head_mode == "per_task":
            return self.heads[str(t)](hidden)
        start = self.class_offset(t)
        return self.shared_head(hidden)[:, start : start + self._task_classes[t]]

    def joined_logits(self, hidden: Tensor) -> Tensor:
        """
        Logits of every class seen so far, in task order: one head over all seen classes,
        as Mamba-CL (joined heads) and Inf-SSM (shared head, columns of seen classes)
        evaluate.

        Args:
            hidden: (B, feature_dim), from extract_hidden().

        Returns:
            (B, total number of classes over all registered tasks)
        """
        self._check_task(None)
        if self.head_mode == "shared":
            return self.shared_head(hidden)[:, : sum(self._task_classes)]
        return torch.cat([self.heads[str(t)](hidden) for t in range(len(self.heads))], dim=-1)

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Trainable backbone parameters (x_proj*, out_proj.weight, A_log, A_b_log), no heads."""
        return [p for p in self.model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _prepare(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected images (B, 3, H, W), got shape {tuple(x.shape)}.")
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
            )
        return x

    def extract_hidden(self, x: Tensor) -> Tensor:
        """
        Pre-logits features of the backbone: model.encode_image(x, pre_logits=True).
        Used by StatsBuffer / NullSpaceProjector during consolidation (wrap in
        torch.no_grad() and eval() there).

        Returns:
            (B, feature_dim)
        """
        hidden = self.model.encode_image(self._prepare(x), pre_logits=True)
        if hidden.ndim != 2 or hidden.shape[-1] != self.feature_dim:
            raise ValueError(
                f"encode_image returned {tuple(hidden.shape)}; expected (B, {self.feature_dim})."
            )
        return hidden

    def forward(self, x: Tensor, task_id: int | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            x:       Images (B, 3, H, W); resized to image_size inside.
            task_id: Task being trained; None = the most recently registered task.

        Returns:
            logits: per_task: (B, classes of task ``task_id``), local labels.
                    shared:   (B, classes of tasks 0..task_id), labels + label_offset().
            hidden: (B, feature_dim)
        """
        t = self._check_task(task_id)
        hidden = self.extract_hidden(x)
        if self.head_mode == "shared":
            return self.shared_head(hidden)[:, : sum(self._task_classes[: t + 1])], hidden
        return self.heads[str(t)](hidden), hidden
