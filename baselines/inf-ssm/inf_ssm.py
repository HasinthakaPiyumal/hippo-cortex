"""Inf-SSM regularizer (Lee et al., CVPR 2026), following vim/cl_methods/inf_ssm.py of the official code.

Each token of each Mamba direction is treated as a diagonal SSM (A, C) with A, C in R^n. Its
extended observability subspace is compared with the old model's via the infinite Grassmannian:

    P = [P11 P12; P21 P22] solves P = A P A^T + C^T C for the stacked system ([A1, A2], [C1, C2]),
    which for diagonal A is P_ij = c_i c_j / (1 - a_i a_j) in O(n^2),
    distance = 1 - tr(P12 P21) / (tr(P11) tr(P22)),

averaged over tokens, then over layers and both scan directions, and scaled by reg_lambda.
"""
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint


def obs_distance(A1: Tensor, C1: Tensor, A2: Tensor, C2: Tensor) -> Tensor:
    """Mean Grassmann distance between rows of (A1, C1) and (A2, C2), all of shape (rows, n).
    Same value as `solve_distance_obs` in the official code (P21 = P12^T, so tr(P12 P21) = ||P12||_F^2)."""
    if torch.allclose(A1, A2) and torch.allclose(C1, C2):  # official shortcut: identical systems -> 0
        return A1.new_zeros(())
    tr11 = (C1 * C1 / (1.0 - A1 * A1)).sum(-1)
    tr22 = (C2 * C2 / (1.0 - A2 * A2)).sum(-1)
    P12 = (C1.unsqueeze(2) * C2.unsqueeze(1)) / (1.0 - A1.unsqueeze(2) * A2.unsqueeze(1))
    return torch.mean(1 - P12.pow(2).sum(dim=(1, 2)) / (tr11 * tr22))


def reg_loss(A1, C1, A2, C2, reg_lambda: float, mode: str = 'inf_ssm') -> Tensor:
    if mode == 'mse':  # ablation: plain MSE on (A, C)
        return reg_lambda * (F.mse_loss(A1, A2) + F.mse_loss(C1, C2))
    if mode == 'inf_ssm':
        return reg_lambda * obs_distance(A1, C1, A2, C2)
    raise ValueError(f"Unknown reg mode {mode}")


def inf_ssm_penalty(states, old_states, reg_lambda: float, mode: str = 'inf_ssm') -> Tensor:
    """states / old_states: lists of (A, C) per layer and direction, from VisionMamba(x, return_states=True)."""
    assert len(states) == len(old_states) > 0
    total = 0
    for (A, C), (A_old, C_old) in zip(states, old_states):
        if torch.is_grad_enabled() and (A.requires_grad or C.requires_grad):
            # recompute in backward: keeps only (A, C), not the (tokens, n, n) Gramian intermediates (~3 GB at batch 64)
            total = total + checkpoint(reg_loss, A, C, A_old, C_old, reg_lambda, mode, use_reentrant=False)
        else:
            total = total + reg_loss(A, C, A_old, C_old, reg_lambda, mode)
    return total / len(states)


class InfSSMForward(nn.Module):
    """Forward pass of one training step: logits of the current model plus the Inf-SSM penalty
    against the frozen old model (zero when there is none, i.e. task 0 or --cl_method seq).

    The penalty is computed inside forward so that under nn.DataParallel each GPU regularizes its
    own slice of the batch: the gathered penalty is one value per GPU, and its mean equals the
    full-batch penalty because the batch is split evenly (drop_last=True, even batch size).
    """

    def __init__(self, net: nn.Module, old_net: nn.Module | None, reg_lambda: float, reg_mode: str):
        super().__init__()
        self.net = net
        self.old_net = old_net
        self.reg_lambda = reg_lambda
        self.reg_mode = reg_mode

    def forward(self, x: Tensor):
        if self.old_net is None:
            return self.net(x), x.new_zeros(1)
        logits, states = self.net(x, return_states=True)
        with torch.no_grad():
            _, old_states = self.old_net(x, return_states=True)
        penalty = inf_ssm_penalty(states, old_states, self.reg_lambda, self.reg_mode)
        return logits, penalty.reshape(1)
