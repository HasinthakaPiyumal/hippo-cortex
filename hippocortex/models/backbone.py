"""
MambaBackbone — De-focus Mamba base encoder + one classification head per task.

OWNER: Praveen Dedigama

This is the same network the Mamba-CL baseline runs (baselines/mamba-cl, config
defocus_mamba_base.yaml, checkpoint defocus_mamba_base_cls.pth), so HippoCortex
and the baseline are directly comparable.

Interface (unchanged for the trainer, plus an optional ``task_id``):
- forward(x, task_id=None)  -> (logits, hidden)
      x:      images (B, 3, H, W); resized to ``image_size`` (224) inside.
      hidden: model.encode_image(x, pre_logits=True), shape (B, feature_dim = 768).
      logits: head of ``task_id`` (default: the most recently added head).
- extract_hidden(x)         -> hidden only. Same extraction point as forward().
- set_task_head(n_classes)  -> adds a head for a new task and returns it. Old heads
                               are kept (and frozen), like Mamba-CL's --seperate_head.

Training scope matches Mamba-CL's --training_string ('head', 'x_proj',
'out_proj.weight', 'A_log'): everything else is frozen.

Loading order matters and mirrors baselines/mamba-cl/train_eval.py:
  build_model -> load_state_dict(strict=False) -> report keys -> reload_params().
reload_params() splits ``x_proj`` into x_proj_step / x_proj_B / x_proj_C, which the
forward pass uses, so it must run after the checkpoint (which stores ``x_proj``) is loaded.

The real model needs mamba-ssm 1.2.2 + causal-conv1d (CUDA). For CPU tests pass a
stand-in via ``MambaBackbone(model=...)`` (see hippocortex/models/fake_defocus.py).
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

# Mamba-CL's --training_string minus 'head': the per-task heads are always trainable
# (current task only), so only the backbone patterns are listed here.
BACKBONE_TRAINABLE_PATTERNS = ("x_proj", "out_proj.weight", "A_log")

# If any of these are missing after load_state_dict the pretrained weights did not load.
REQUIRED_KEY_PREFIXES = ("blocks.", "patch_embed.", "norm.", "cls_token")


def _load_mamba_cl_config_module() -> ModuleType:
    """Import baselines/mamba-cl/utils as a private package and return its ``config`` module."""
    cfg_name = f"{_PKG}.config"
    if cfg_name in sys.modules:
        return sys.modules[cfg_name]

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
        return importlib.import_module(cfg_name)
    except BaseException:
        for name in [n for n in sys.modules if n == _PKG or n.startswith(_PKG + ".")]:
            del sys.modules[name]
        raise


def check_load_report(missing: list[str], unexpected: list[str]) -> None:
    """Print the load_state_dict report and raise if the backbone weights did not load."""
    print(f":: loaded defocus_mamba_base: missing keys {missing}, unexpected keys {unexpected}")
    bad = [k for k in missing if k.startswith(REQUIRED_KEY_PREFIXES)]
    if bad:
        raise RuntimeError(
            f"Pretrained weights did not load: {len(bad)} required keys are missing "
            f"(e.g. {bad[:4]}). Check the checkpoint file and the model config."
        )


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
    check_load_report(list(info.missing_keys), list(info.unexpected_keys))
    model.reload_params()
    return model


class MambaBackbone(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        config_path: str | Path | None = None,
        model: nn.Module | None = None,
        image_size: int = 224,
    ) -> None:
        """
        Args:
            checkpoint_path: defocus_mamba_base_cls.pth. Required unless ``model`` is given.
            config_path:     De-focus yaml; defaults to baselines/mamba-cl/utils/defocus_mamba_base.yaml.
            model:           Pre-built network exposing ``embed_dim`` and
                             ``encode_image(x, pre_logits=True)``. Used by tests; skips
                             building and loading the real model.
            image_size:      Inputs are bicubic-resized to image_size x image_size.
        """
        super().__init__()
        if model is None:
            if checkpoint_path is None:
                raise ValueError("Pass checkpoint_path (real model) or model (stand-in).")
            model = _build_defocus_model(
                Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH,
                Path(checkpoint_path),
            )

        self.image_size = image_size
        self.feature_dim = int(model.embed_dim)
        # The 1000-class ImageNet head is replaced by per-task heads below.
        model.head = nn.Identity()
        self.model = model
        self.heads = nn.ModuleDict()

        for name, p in self.model.named_parameters():
            p.requires_grad_(any(pat in name for pat in BACKBONE_TRAINABLE_PATTERNS))

    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------

    @property
    def n_tasks(self) -> int:
        """Number of task heads created so far."""
        return len(self.heads)

    def set_task_head(self, n_classes: int, task_id: int | None = None) -> nn.Linear:
        """
        Add a classification head for a new task; earlier heads are kept and frozen.

        Args:
            n_classes: Output classes of the new head.
            task_id:   Task index. None appends as the next task. An existing index
                       re-creates that head (re-running a task); a gap is an error.

        Returns:
            The new head. Its parameters are NOT in any previously built optimizer;
            the caller must build or extend the optimizer with them.
        """
        if task_id is None:
            task_id = len(self.heads)
        if not 0 <= task_id <= len(self.heads):
            raise ValueError(f"task_id={task_id} leaves a gap; {len(self.heads)} heads exist.")

        ref = next(self.model.parameters())
        head = nn.Linear(self.feature_dim, n_classes).to(device=ref.device, dtype=ref.dtype)
        self.heads[str(task_id)] = head

        for key, h in self.heads.items():
            h.requires_grad_(key == str(task_id))
            if key != str(task_id):
                # No optimizer owns a frozen head, so nothing would ever clear its old grads.
                for p in h.parameters():
                    p.grad = None
        return head

    def _head(self, task_id: int | None) -> nn.Linear:
        if len(self.heads) == 0:
            raise RuntimeError("No task head yet; call set_task_head() first.")
        key = str(len(self.heads) - 1 if task_id is None else task_id)
        if key not in self.heads:
            raise KeyError(f"No head for task {task_id}; have {list(self.heads.keys())}.")
        return self.heads[key]

    def head_parameters(self, task_id: int | None = None) -> list[nn.Parameter]:
        return list(self._head(task_id).parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Trainable backbone parameters (x_proj*, out_proj.weight, A_log), excluding heads."""
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
            task_id: Head to use; None = most recently added head.

        Returns:
            logits: (B, n_classes_of_head)
            hidden: (B, feature_dim)
        """
        hidden = self.extract_hidden(x)
        return self._head(task_id)(hidden), hidden
