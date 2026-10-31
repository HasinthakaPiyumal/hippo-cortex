"""Vision Mamba (Vim-small) backbone for the Inf-SSM baseline.

Port of the model used by the official Inf-SSM code (https://github.com/isaacNingLee/Inf-SSM):
`vim/models_mamba.py` (from hustvl/Vim) and the bidirectional `v2_state_ac` mixer in
`mamba-1p1p1/mamba_ssm/modules/mamba_simple.py` + `ops/selective_scan_interface.py`.

The official code runs on a fork of mamba-ssm 1.1.1 with fused Vim kernels. Here the same
computation is assembled from the public kernels of upstream mamba-ssm / causal-conv1d
(`selective_scan_fn`, `causal_conv1d_fn`), so it runs on the stock mamba-ssm 1.2.2 install used
for Mamba-CL. Pure-PyTorch fallbacks are used when the kernels are missing or the tensors are on
CPU, so the model can be unit-tested without a GPU. Parameter names match the official
checkpoints, so the released task-0 checkpoints and the hustvl Vim weights load as-is.
"""
import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers import DropPath, trunc_normal_, lecun_normal_
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

import abar_triton

KERNEL_IMPORT_ERRORS = {}  # why a kernel is unavailable, for the Kaggle setup check
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError as e:  # also raised when mamba_ssm's own transformers import fails
    selective_scan_fn = None
    KERNEL_IMPORT_ERRORS['mamba_ssm'] = repr(e)
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError as e:
    causal_conv1d_fn = None
    KERNEL_IMPORT_ERRORS['causal_conv1d'] = repr(e)


# ---------------------------------------------------------------- kernels and their fallbacks

def selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus=True):
    """Reference selective scan (mamba_ssm.ops.selective_scan_interface.selective_scan_ref), real A, variable B/C.
    u, delta, z: (b, d, l); A: (d, n); B, C: (b, n, l); D, delta_bias: (d,). Returns (b, d, l)."""
    dtype_in = u.dtype
    u, delta = u.float(), delta.float()
    delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
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


def selective_scan(u, delta, A, B, C, D, z, delta_bias):
    if selective_scan_fn is not None and u.is_cuda:
        return selective_scan_fn(u, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias)


def causal_conv_silu(x, weight, bias):
    """Depthwise causal conv1d + SiLU. x: (b, d, l); weight: (d, 1, w) as stored by nn.Conv1d."""
    if causal_conv1d_fn is not None and x.is_cuda:
        return causal_conv1d_fn(x, rearrange(weight, "d 1 w -> d w"), bias, activation="silu")
    seqlen = x.shape[-1]
    return F.silu(F.conv1d(x, weight, bias, padding=weight.shape[-1] - 1, groups=x.shape[1])[..., :seqlen])


def _upcast(x: Tensor) -> Tensor:
    """fp16/bf16 -> fp32; fp32/fp64 unchanged."""
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _abar_fwd_torch(delta: Tensor, A: Tensor) -> Tensor:
    """Reference A_bar, looped over n so the (b, d, l, n) tensor is never held in memory."""
    sp = F.softplus(_upcast(delta))
    return torch.stack([torch.exp(sp * A[:, n, None]).mean(dim=1) for n in range(A.shape[1])], dim=-1)


def _abar_bwd_torch(delta: Tensor, A: Tensor, grad: Tensor):
    d32 = _upcast(delta)
    sp = F.softplus(d32)
    n_inner = delta.shape[1]
    grad = _upcast(grad)
    g_delta = torch.zeros_like(d32)
    g_A = torch.zeros_like(A)
    for n in range(A.shape[1]):
        e = torch.exp(sp * A[:, n, None]) * grad[:, None, :, n]  # (b, d, l)
        g_A[:, n] = (e * sp).sum(dim=(0, 2)) / n_inner
        g_delta += e * A[:, n, None]
    return g_delta * torch.sigmoid(d32) / n_inner, g_A


def _use_triton(delta: Tensor, A: Tensor) -> bool:
    return (delta.is_cuda and abar_triton.triton is not None and delta.dtype != torch.float64 and A.dtype == torch.float32
            and delta.stride(-1) == 1 and abar_triton.available(_abar_fwd_torch, _abar_bwd_torch, delta.device))


class _MeanDiscreteA(torch.autograd.Function):
    """A_bar[b, l, n] = mean_d exp(softplus(delta[b, d, l]) * A[d, n]).

    Same value and gradients as `A_bar` in the official `MambaInnerFnNoOutProj_state`. As in the official
    code, `delta` is the dt projection *before* `dt_proj.bias` is added (the scan kernel itself does add
    the bias). On GPU it uses the fused Triton kernels of abar_triton.py once they have been checked
    against the PyTorch version below.
    """

    @staticmethod
    def forward(ctx, delta: Tensor, A: Tensor) -> Tensor:
        ctx.save_for_backward(delta, A)
        if _use_triton(delta, A):
            return abar_triton.abar_fwd(delta, A)
        return _abar_fwd_torch(delta, A)

    @staticmethod
    def backward(ctx, grad: Tensor):
        delta, A = ctx.saved_tensors
        if _use_triton(delta, A):
            g_delta, g_A = abar_triton.abar_bwd(delta, A, grad)
        else:
            g_delta, g_A = _abar_bwd_torch(delta, A, grad)
        return g_delta.to(delta.dtype), g_A


def soft_normalize(x: Tensor) -> Tensor:
    """SN(x) = 2 / (1 + exp(-x)) - 1, applied to the states to enforce Schur stability (paper Sec. 4)."""
    return 2 * torch.sigmoid(x) - 1


# ---------------------------------------------------------------- bidirectional Mamba mixer

class BiMamba(nn.Module):
    """Vim's bidirectional Mamba mixer (bimamba_type="v2", if_divide_out=True).

    With `return_states=True` the forward also returns the per-token state pairs that Inf-SSM
    regularizes, exactly like the official `v2_state_ac` path: for each direction,
    A = SN(mean over the inner dim of exp(softplus(delta) A)) and C = SN(C), both reshaped to
    (batch * seqlen, d_state), one row per token.
    """

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
            setattr(self, f"A_log" if sfx == "" else "A_b_log", A_log)
            D = nn.Parameter(torch.ones(self.d_inner))
            D._no_weight_decay = True
            setattr(self, f"D{sfx}", D)

    def _scan(self, xz: Tensor, conv1d: nn.Conv1d, x_proj: nn.Linear, dt_proj: nn.Linear, A: Tensor, D: Tensor,
              return_states: bool):
        """One direction of MambaInnerFnNoOutProj(_state). xz: (b, 2*d_inner, l) -> y: (b, d_inner, l)."""
        seqlen = xz.shape[-1]
        x, z = xz.chunk(2, dim=1)
        x = causal_conv_silu(x, conv1d.weight, conv1d.bias)
        x_dbl = x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = rearrange(dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=seqlen)  # bias is added inside the scan
        B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()
        y = selective_scan(x, delta, A, B, C, D.float(), z, dt_proj.bias.float())
        if not return_states:
            return y, None
        A_bar = _MeanDiscreteA.apply(delta, A)  # (b, l, n)
        A_r = soft_normalize(A_bar.reshape(-1, self.d_state))
        C_r = soft_normalize(_upcast(C.transpose(1, 2).reshape(-1, self.d_state)))
        return y, (A_r, C_r)

    def forward(self, hidden_states: Tensor, return_states: bool = False):
        """hidden_states: (b, l, d_model). Returns (out, states) with states = [(A_f, C_f), (A_b, C_b)] or None."""
        seqlen = hidden_states.shape[1]
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"), "d (b l) -> b d l", l=seqlen)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_b_log.float())
        y, s_f = self._scan(xz, self.conv1d, self.x_proj, self.dt_proj, A, self.D, return_states)
        y_b, s_b = self._scan(xz.flip([-1]), self.conv1d_b, self.x_proj_b, self.dt_proj_b, A_b, self.D_b, return_states)
        y = y + y_b.flip([-1])
        if self.if_divide_out:
            y = y / 2
        out = F.linear(rearrange(y, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        return out, ([s_f, s_b] if return_states else None)


# ---------------------------------------------------------------- Vim backbone

class RMSNorm(nn.Module):
    """RMSNorm with the fused add+norm semantics of mamba_ssm.ops.triton.layernorm (pure PyTorch).
    Holds only `weight`, like the Triton module, so checkpoint keys match."""

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

    def forward(self, hidden_states: Tensor, residual: Tensor | None, return_states: bool = False):
        if residual is not None:
            hidden_states = self.drop_path(hidden_states)
        hidden_states, residual = self.norm(hidden_states, residual=residual, prenorm=True, residual_in_fp32=True)
        hidden_states, states = self.mixer(hidden_states, return_states=return_states)
        return hidden_states, residual, states


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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    def forward_features(self, x: Tensor, return_states: bool = False):
        x = self.patch_embed(x)
        B, M, _ = x.shape
        token_position = M // 2  # class token in the middle of the patch sequence
        x = torch.cat((x[:, :token_position], self.cls_token.expand(B, -1, -1), x[:, token_position:]), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        residual, hidden_states, states = None, x, []
        for layer in self.layers:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                hidden_states, residual, s = checkpoint(layer, hidden_states, residual, return_states, use_reentrant=False)
            else:
                hidden_states, residual, s = layer(hidden_states, residual, return_states)
            if return_states:
                states.extend(s)
        hidden_states = self.norm_f(self.drop_path(hidden_states), residual=residual, prenorm=False)
        return hidden_states[:, token_position, :], (states if return_states else None)

    def forward(self, x: Tensor, return_states: bool = False):
        """Returns logits, or (logits, states) with states = [(A, C)] for every layer and direction (2 * depth pairs)."""
        feats, states = self.forward_features(x, return_states)
        logits = self.head(feats)
        return (logits, states) if return_states else logits


def vim_small(num_classes=1000, drop_path_rate=0.1, img_size=224, **kwargs) -> VisionMamba:
    """Vim-small (26M params): embed_dim 384, depth 24, d_state 16."""
    kwargs.setdefault('depth', 24)
    return VisionMamba(patch_size=16, embed_dim=384, num_classes=num_classes, drop_path_rate=drop_path_rate, img_size=img_size, **kwargs)


def vim_tiny(num_classes=1000, drop_path_rate=0.1, img_size=224, **kwargs) -> VisionMamba:
    """Vim-tiny (7M params): embed_dim 192, depth 24, d_state 16."""
    kwargs.setdefault('depth', 24)
    return VisionMamba(patch_size=16, embed_dim=192, num_classes=num_classes, drop_path_rate=drop_path_rate, img_size=img_size, **kwargs)


MODELS = {"vim_small": vim_small, "vim_tiny": vim_tiny}


def load_finetune(model: VisionMamba, path: str) -> None:
    """Load a Vim checkpoint the way the official main.py does for --finetune: drop a head whose
    shape differs (e.g. the 1000-class ImageNet head), interpolate pos_embed, load non-strictly."""
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
    print(f":: loaded {path}: {len(state)} tensors, missing={missing}, unexpected={unexpected}")
