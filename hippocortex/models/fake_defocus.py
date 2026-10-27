"""
TinyDefocus — a small CPU stand-in for the De-focus Mamba base network.

The real model (baselines/mamba-cl) cannot be constructed without the CUDA
kernels from mamba-ssm / causal-conv1d, so unit tests and the CPU dry-run of
scripts/smoke_defocus_backbone.py inject this instead via
``MambaBackbone(model=TinyDefocus(...))``.

It copies the parts of the real interface that MambaBackbone and the Trainer
depend on, and nothing else:
  - ``embed_dim`` and ``encode_image(x, pre_logits=True) -> (B, embed_dim)``;
  - parameter names the training patterns match on
    (``x_proj_step/B/C``, ``out_proj.weight``, ``A_log``) plus frozen ones
    (``patch_embed.proj``, ``in_proj``, ``norm``, ``cls_token``);
  - a 1000-way-style ``head`` that MambaBackbone must replace;
  - ``reload_params()``, which the real model needs after load_state_dict.

Every trainable parameter receives a gradient, so gradient-flow tests are meaningful.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, L, D)


class _TinyMixer(nn.Module):
    def __init__(self, d_model: int, d_inner: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.x_proj_step = nn.Linear(d_inner, dt_rank, bias=False)
        self.x_proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.x_proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))

    def forward(self, x: Tensor) -> Tensor:
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = u + u.mean(dim=1, keepdim=True)  # global token mixing, so the cls token sees the image
        step = self.x_proj_step(u).sum(-1, keepdim=True)
        b = self.x_proj_B(u)
        c = self.x_proj_C(u).sum(-1, keepdim=True)
        a = torch.exp(self.A_log)  # (d_inner, d_state)
        y = u * torch.sigmoid(gate) * (1 + torch.tanh(step)) + torch.sigmoid(c) * (b @ a.T)
        return self.out_proj(y)


class _TinyBlock(nn.Module):
    def __init__(self, d_model: int, d_inner: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = _TinyMixer(d_model, d_inner, d_state, dt_rank)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mixer(self.norm(x))


class TinyDefocus(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        depth: int = 2,
        d_inner: int = 64,
        d_state: int = 4,
        dt_rank: int = 3,
        patch_size: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_TinyBlock(embed_dim, d_inner, d_state, dt_rank) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.reload_calls = 0

    def reload_params(self) -> None:
        self.reload_calls += 1

    def encode_image(
        self, x: Tensor, drop_path_rate: float = 0.0, tblogger=None, pre_logits: bool = True
    ) -> Tensor:
        assert pre_logits
        tokens = self.patch_embed(x)
        tokens = torch.cat([tokens, self.cls_token.expand(tokens.shape[0], -1, -1)], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)[:, -1]

    def forward(self, x: Tensor, drop_path_rate: float = 0.0, tblogger=None) -> Tensor:
        return self.head(self.encode_image(x))
