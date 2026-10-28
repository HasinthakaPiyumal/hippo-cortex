"""Vision Mamba (Vim-small) backbone for the Mamba-CL baseline.

Same network as baselines/inf-ssm/mamba_vim.py (hustvl/Vim, bimamba v2, middle class token,
if_divide_out), with the same parameter names, so the hustvl Vim-small weights load as-is. It is
adapted to what Mamba-CL's null-space code needs from a model, mirroring utils/mamba_block.MambaMixer:

- split_parameters() (called by reload_params(), after the checkpoint is loaded) splits each
  direction's x_proj into x_proj{sfx}_step / _B / _C and attaches one IntermReader per projected
  parameter. Vim is bidirectional, so a block has 4 readers per direction (step, B, C, A) plus one
  for out_proj: 9 projected parameters per block.
- delta = softplus(dt_proj(dt) + dt_bias) is computed outside the scan kernel (delta_softplus=False),
  so the A and B readers see the real step size, as in MambaMixer.forward_ssm.
- encode_image(x, pre_logits=True), embed_dim and head match DefocusAttentionNetwork.

The backward direction's A is stored as A_b_log, which Mamba-CL's default 'A_log' pattern does not
match; train_eval.py adds 'A_b_log' to --training_string and --null_patterns for Vim models.

Pure-PyTorch fallbacks are used when the kernels are missing or the tensors are on CPU, so the model
can be unit-tested without a GPU.
"""
import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_, lecun_normal_
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .interm_reader import IntermReader

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

VIM_SMALL_IN1K = ("hustvl/Vim-small-midclstok", "vim_s_midclstok_80p5acc.pth")  # --pretrained_path vim_small_in1k


# ---------------------------------------------------------------- kernels and their fallbacks

def selective_scan_ref(u, delta, A, B, C, D, z):
    """Reference selective scan with a precomputed delta (no delta_bias, no softplus).
    u, delta, z: (b, d, l); A: (d, n); B, C: (b, n, l); D: (d,). Returns (b, d, l)."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B.float(), u)
    x = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum('bdn,bn->bd', x, C[:, :, i].float()))
    y = torch.stack(ys, dim=2) + u * D[None, :, None]
    y = y * F.silu(z.float())
    return y.to(dtype_in)


def selective_scan(u, delta, A, B, C, D, z):
    if selective_scan_fn is not None and u.is_cuda:
        return selective_scan_fn(u, delta, A, B, C, D, z=z, delta_bias=None, delta_softplus=False)
    return selective_scan_ref(u, delta, A, B, C, D, z)


def causal_conv_silu(x, weight, bias):
    """Depthwise causal conv1d + SiLU. x: (b, d, l); weight: (d, 1, w) as stored by nn.Conv1d."""
    if causal_conv1d_fn is not None and x.is_cuda:
        return causal_conv1d_fn(x, rearrange(weight, "d 1 w -> d w"), bias, activation="silu")
    seqlen = x.shape[-1]
    return F.silu(F.conv1d(x, weight, bias, padding=weight.shape[-1] - 1, groups=x.shape[1])[..., :seqlen])


# ---------------------------------------------------------------- mixer and block

class BiMamba(nn.Module):
    """Vim's bidirectional Mamba mixer (bimamba_type="v2", if_divide_out=True) with Mamba-CL's
    split x_proj and IntermReaders. forward() needs split_parameters() to have run."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False,
                 if_divide_out=True, layer_idx=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.if_divide_out = if_divide_out
        self.layer_idx = layer_idx
        self.is_split = False

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        # forward direction: conv1d, x_proj, dt_proj, A_log, D; backward direction: the same with "_b"
        for sfx in ("", "_b"):
            setattr(self, f"conv1d{sfx}", nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                                     padding=d_conv - 1, bias=conv_bias))
            setattr(self, f"x_proj{sfx}", nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False))
            dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            # dt init from mamba_simple.py: softplus(dt_bias) in [dt_min, dt_max]
            dt_init_std = self.dt_rank ** -0.5 * dt_scale
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            else:
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
            with torch.no_grad():
                dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            dt_proj.bias._no_reinit = True
            setattr(self, f"dt_proj{sfx}", dt_proj)
            A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
            A_log = nn.Parameter(torch.log(A))
            A_log._no_weight_decay = True
            setattr(self, self._a_name(sfx), A_log)
            D = nn.Parameter(torch.ones(self.d_inner))
            D._no_weight_decay = True
            setattr(self, f"D{sfx}", D)

    @staticmethod
    def _a_name(sfx: str) -> str:
        return "A_log" if sfx == "" else "A_b_log"  # the checkpoint names

    def split_parameters(self):
        """Split x_proj{sfx} into x_proj{sfx}_step / _B / _C and attach the IntermReaders
        (as MambaMixer.split_parameters). Run once, after the checkpoint is loaded."""
        assert not self.is_split, "split_parameters() already ran"
        for sfx in ("", "_b"):
            x_proj: nn.Linear = getattr(self, f"x_proj{sfx}")
            parts = torch.split(x_proj.weight.data, [self.dt_rank, self.d_state, self.d_state], dim=0)
            for name, w in zip(("step", "B", "C"), parts):
                lin = nn.Linear(self.d_inner, w.shape[0], bias=False, device=w.device, dtype=w.dtype)
                lin.weight.data = w.clone()
                setattr(self, f"x_proj{sfx}_{name}", lin)
                setattr(self, f"interm_reader{sfx}_{name}", IntermReader(id(lin.weight), f"interm_reader_{name}"))
            delattr(self, f"x_proj{sfx}")
            setattr(self, f"interm_reader{sfx}_A", IntermReader(id(getattr(self, self._a_name(sfx))), "interm_reader_A"))
        self.interm_reader_out = IntermReader(id(self.out_proj.weight), "interm_reader_out")
        self.is_split = True

    def _scan(self, xz: Tensor, sfx: str) -> Tensor:
        """One direction. xz: (b, 2*d_inner, l) -> y: (b, d_inner, l)."""
        x, z = xz.chunk(2, dim=1)
        conv1d: nn.Conv1d = getattr(self, f"conv1d{sfx}")
        x = causal_conv_silu(x, conv1d.weight, conv1d.bias)
        u = rearrange(x, "b d l -> b l d")
        x_proj_B: nn.Linear = getattr(self, f"x_proj{sfx}_B")

        dt = getattr(self, f"x_proj{sfx}_step")(getattr(self, f"interm_reader{sfx}_step")(u))  # (b, l, dt_rank)
        B = x_proj_B(u)
        C = getattr(self, f"x_proj{sfx}_C")(getattr(self, f"interm_reader{sfx}_C")(u))

        dt_proj: nn.Linear = getattr(self, f"dt_proj{sfx}")
        # delta outside the kernel, so the readers below see the real step size (the scan dtype must match u)
        delta = F.softplus((dt_proj.weight @ dt.transpose(1, 2)).float() + dt_proj.bias.float()[:, None]).to(x.dtype)  # (b, d, l)

        if not x_proj_B.weight.requires_grad:  # null-space collection runs with every parameter frozen
            getattr(self, f"interm_reader{sfx}_A")(delta)
            getattr(self, f"interm_reader{sfx}_B")(torch.matmul(delta, u))

        A = -torch.exp(getattr(self, self._a_name(sfx)).float())
        B = rearrange(B, "b l n -> b n l").contiguous()
        C = rearrange(C, "b l n -> b n l").contiguous()
        return selective_scan(x, delta, A, B, C, getattr(self, f"D{sfx}").float(), z)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """hidden_states: (b, l, d_model) -> (b, l, d_model)."""
        assert self.is_split, "call model.reload_params() after loading the checkpoint"
        seqlen = hidden_states.shape[1]
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        y = self._scan(xz, "")
        y_b = self._scan(xz.flip([-1]), "_b")
        y = y + y_b.flip([-1])
        if self.if_divide_out:
            y = y / 2
        return self.out_proj(self.interm_reader_out(rearrange(y, "b d l -> b l d")))


class RMSNorm(nn.Module):
    """RMSNorm with the fused add+norm semantics of mamba_ssm.ops.triton.layernorm (pure PyTorch)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, residual: Tensor | None = None, prenorm: bool = False, residual_in_fp32: bool = True):
        r = x.float() if residual is None else residual.float() + x.float()
        out = (r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)
        if not prenorm:
            return out
        return out, (r if residual_in_fp32 else r.to(x.dtype))


class Block(nn.Module):
    """Add -> RMSNorm -> BiMamba (fused_add_norm=True, residual_in_fp32=True in Vim-small)."""

    def __init__(self, dim, layer_idx, drop_path=0., d_state=16, if_divide_out=True, norm_epsilon=1e-5):
        super().__init__()
        self.mixer = BiMamba(dim, d_state=d_state, if_divide_out=if_divide_out, layer_idx=layer_idx)
        self.norm = RMSNorm(dim, eps=norm_epsilon)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_idx = layer_idx

    def forward(self, hidden_states: Tensor, residual: Tensor | None):
        if residual is not None:
            hidden_states = self.drop_path(hidden_states)
        hidden_states, residual = self.norm(hidden_states, residual=residual, prenorm=True, residual_in_fp32=True)
        return self.mixer(hidden_states), residual


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.grid_size = ((img_size - patch_size) // stride + 1,) * 2
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.Identity()

    def forward(self, x):
        assert x.shape[-2:] == self.img_size, f"input size {tuple(x.shape[-2:])} != model size {self.img_size}"
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


def _segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _mamba_init_weights(module, n_layer):
    if isinstance(module, nn.Linear) and module.bias is not None and not getattr(module.bias, "_no_reinit", False):
        nn.init.zeros_(module.bias)
    for name, p in module.named_parameters():
        if name == "out_proj.weight":  # GPT-2 style rescale of the residual projections
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p /= math.sqrt(n_layer)


# ---------------------------------------------------------------- Vim backbone

class VisionMamba(nn.Module):
    """Vim with a middle class token, absolute position embedding and bidirectional v2 mixers
    (`vim_*_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2`)."""

    def __init__(self, img_size=224, patch_size=16, depth=24, embed_dim=384, d_state=16, num_classes=1000,
                 drop_rate=0., drop_path_rate=0.1, norm_epsilon=1e-5, grad_ckpt=False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = self.num_features = embed_dim
        self.grad_ckpt = grad_ckpt
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, stride=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr  # as in Vim: layer i gets inter_dpr[i]
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.layers = nn.ModuleList([Block(embed_dim, layer_idx=i, drop_path=inter_dpr[i], d_state=d_state,
                                           norm_epsilon=norm_epsilon) for i in range(depth)])
        self.norm_f = RMSNorm(embed_dim, eps=norm_epsilon)

        self.patch_embed.apply(_segm_init_weights)
        self.head.apply(_segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(partial(_mamba_init_weights, n_layer=depth))

    def reload_params(self):
        """As DefocusAttentionNetwork.reload_params: split x_proj and attach the IntermReaders."""
        for layer in self.layers:
            layer.mixer.split_parameters()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        B, M, _ = x.shape
        token_position = M // 2  # class token in the middle of the patch sequence
        x = torch.cat((x[:, :token_position], self.cls_token.expand(B, -1, -1), x[:, token_position:]), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        residual, hidden_states = None, x
        for layer in self.layers:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                hidden_states, residual = checkpoint(layer, hidden_states, residual, use_reentrant=False)
            else:
                hidden_states, residual = layer(hidden_states, residual)
        hidden_states = self.norm_f(self.drop_path(hidden_states), residual=residual, prenorm=False)
        return hidden_states[:, token_position, :]

    def encode_image(self, x: Tensor, pre_logits: bool = True) -> Tensor:
        assert pre_logits
        return self.forward_features(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x))


def vim_small(num_classes=1000, drop_path_rate=0.1, img_size=224, **kwargs) -> VisionMamba:
    """Vim-small (26M params): embed_dim 384, depth 24, d_state 16."""
    kwargs.setdefault('depth', 24)
    return VisionMamba(patch_size=16, embed_dim=384, num_classes=num_classes, drop_path_rate=drop_path_rate, img_size=img_size, **kwargs)


MODELS = {"vim_small": vim_small}


def load_finetune(model: VisionMamba, path: str) -> tuple[list[str], list[str]]:
    """Load a Vim checkpoint the way the official main.py does for --finetune: drop a head whose
    shape differs, interpolate pos_embed, load non-strictly. Must run before reload_params()."""
    assert not any(layer.mixer.is_split for layer in model.layers), "load the checkpoint before reload_params()"
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    own = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in state and k in own and state[k].shape != own[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del state[k]
    if 'pos_embed' in state:
        pos = state['pos_embed']
        num_extra = model.pos_embed.shape[-2] - model.patch_embed.num_patches
        orig_size = int((pos.shape[-2] - num_extra) ** 0.5)
        new_size = int(model.patch_embed.num_patches ** 0.5)
        if orig_size != new_size:
            extra, tokens = pos[:, :num_extra], pos[:, num_extra:]
            tokens = tokens.reshape(-1, orig_size, orig_size, pos.shape[-1]).permute(0, 3, 1, 2)
            tokens = F.interpolate(tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            state['pos_embed'] = torch.cat((extra, tokens.permute(0, 2, 3, 1).flatten(1, 2)), dim=1)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)
