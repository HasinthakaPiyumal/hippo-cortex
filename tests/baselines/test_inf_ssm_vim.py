"""CPU tests for the Inf-SSM baseline (baselines/inf-ssm): the regularizer and the Vim state
extraction are checked against the formulas of the official code (isaacNingLee/Inf-SSM)."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

INF_SSM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../baselines/inf-ssm"))
if INF_SSM_DIR not in sys.path:
    sys.path.insert(0, INF_SSM_DIR)

pytest.importorskip("timm")
from inf_ssm import InfSSMForward, inf_ssm_penalty, obs_distance  # noqa: E402
from mamba_vim import BiMamba, _MeanDiscreteA, vim_small, vim_tiny  # noqa: E402


# ---- official reference (vim/cl_methods/inf_ssm.py), copied verbatim
def _official_solve_P(A1, C1, A2, C2):
    A_diag = torch.cat([A1, A2], dim=1)
    denom = 1.0 - A_diag.unsqueeze(2) * A_diag.unsqueeze(1)
    C_combined = torch.cat([C1, C2], dim=1)
    Q = C_combined.unsqueeze(2) * C_combined.unsqueeze(1)
    return Q / denom


def _official_distance(A1, C1, A2, C2):
    N = A1.shape[1]
    P = _official_solve_P(A1, C1, A2, C2)
    tr = lambda m: torch.einsum('bii->b', m)  # noqa: E731
    return torch.mean(1 - tr(P[:, :N, N:] @ P[:, N:, :N]) / (tr(P[:, :N, :N]) * tr(P[:, N:, N:])))


def _rand_states(rows=50, n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = 2 * torch.sigmoid(torch.rand(rows, n, generator=g)) - 1  # (0, 0.46) like the real states
    C = 2 * torch.sigmoid(torch.randn(rows, n, generator=g)) - 1
    return A.double(), C.double()


def test_distance_matches_official():
    A1, C1 = _rand_states(seed=0)
    A2, C2 = _rand_states(seed=1)
    assert torch.allclose(obs_distance(A1, C1, A2, C2), _official_distance(A1, C1, A2, C2), atol=1e-12)


def test_distance_zero_for_same_system_and_in_unit_interval():
    A1, C1 = _rand_states(seed=0)
    A2, C2 = _rand_states(seed=1)
    assert obs_distance(A1, C1, A1, C1).item() == 0.0  # official allclose shortcut
    d = obs_distance(A1, C1, A2, C2)
    assert 0.0 < d.item() < 1.0
    # invariant to rescaling either C (the observability subspace does not change)
    assert torch.allclose(obs_distance(A1, C1, A2, 3.0 * C2), d, atol=1e-12)
    assert torch.allclose(obs_distance(A1, -0.5 * C1, A2, C2), d, atol=1e-12)


def test_gramian_solves_sylvester():
    """P12 = C1 C2^T / (1 - a1 a2^T) solves P = A1 P A2 + C1^T C2 for diagonal A (one row = one SSM)."""
    A1, C1 = _rand_states(rows=1, seed=0)
    A2, C2 = _rand_states(rows=1, seed=1)
    P = (C1[0][:, None] * C2[0][None, :]) / (1 - A1[0][:, None] * A2[0][None, :])
    rhs = torch.diag(A1[0]) @ P @ torch.diag(A2[0]) + C1[0][:, None] @ C2[0][None, :]
    assert torch.allclose(P, rhs, atol=1e-12)


def test_penalty_averages_layers_and_scales_lambda():
    s_old = [_rand_states(seed=i) for i in range(4)]
    s_new = [_rand_states(seed=10 + i) for i in range(4)]
    expect = sum(_official_distance(*n, *o) for n, o in zip(s_new, s_old)) / 4 * 123.0
    assert torch.allclose(inf_ssm_penalty(s_new, s_old, 123.0), expect)


def test_mean_discrete_A_matches_autograd():
    torch.manual_seed(0)
    delta = torch.randn(2, 6, 5, dtype=torch.float64, requires_grad=True)
    A = -torch.rand(6, 4, dtype=torch.float64).add(0.5).requires_grad_(True)
    ref = torch.exp(torch.einsum('bdl,dn->bdln', F.softplus(delta), A)).mean(dim=1)  # official formula, (b, l, n)
    out = _MeanDiscreteA.apply(delta, A)
    assert torch.allclose(out, ref)
    assert torch.autograd.gradcheck(_MeanDiscreteA.apply, (delta, A))


def test_state_extraction_layout():
    """States are (A_bar, C) per token, rows ordered (batch, position), soft-normalised, for both directions."""
    torch.manual_seed(0)
    mixer = BiMamba(d_model=8, d_state=4).double()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    out, states = mixer(x, return_states=True)
    assert out.shape == (2, 5, 8) and len(states) == 2
    A_r, C_r = states[0]
    assert A_r.shape == C_r.shape == (2 * 5, 4)

    # recompute the forward direction by hand, following MambaInnerFnNoOutProj_state
    xz = (x @ mixer.in_proj.weight.t()).transpose(1, 2)
    xc = F.silu(F.conv1d(xz[:, :16], mixer.conv1d.weight, mixer.conv1d.bias, padding=3, groups=16)[..., :5])
    x_dbl = mixer.x_proj(xc.transpose(1, 2).reshape(-1, 16))
    dt, _, C = torch.split(x_dbl, [mixer.dt_rank, 4, 4], dim=-1)
    delta = (mixer.dt_proj.weight @ dt.t()).reshape(16, 2, 5).permute(1, 0, 2)
    A = -torch.exp(mixer.A_log)
    A_bar = torch.exp(torch.einsum('bdl,dn->bdln', F.softplus(delta), A)).mean(dim=1)
    assert torch.allclose(A_r, 2 * torch.sigmoid(A_bar.reshape(-1, 4)) - 1)
    assert torch.allclose(C_r, 2 * torch.sigmoid(C) - 1)
    assert 0 < A_r.min() and A_r.max() < 1  # A_bar in (0, 1) -> SN(A_bar) in (0, 0.47): Schur stable


def test_vim_small_shapes_and_param_count():
    m = vim_small(num_classes=100, drop_path_rate=0.0)
    n = sum(p.numel() for p in m.parameters())
    assert 25e6 < n < 27e6  # Vim-small: 26M
    keys = m.state_dict().keys()
    for k in ('layers.0.mixer.A_b_log', 'layers.23.mixer.x_proj_b.weight', 'layers.0.norm.weight', 'norm_f.weight', 'cls_token'):
        assert k in keys


def test_reg_step_backprops_into_current_model_only():
    torch.manual_seed(0)
    net = vim_tiny(num_classes=10, drop_path_rate=0.0, img_size=32, depth=2)
    old = vim_tiny(num_classes=10, drop_path_rate=0.0, img_size=32, depth=2)
    old.load_state_dict(net.state_dict())
    old.eval().requires_grad_(False)
    with torch.no_grad():
        net.layers[1].mixer.x_proj.weight.add_(0.05 * torch.randn_like(net.layers[1].mixer.x_proj.weight))
    fwd = InfSSMForward(net, old, reg_lambda=1e3, reg_mode='inf_ssm')
    logits, penalty = fwd(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 10) and penalty.shape == (1,) and penalty.item() > 0
    penalty.sum().backward()
    assert net.layers[1].mixer.x_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in old.parameters())


def test_grad_checkpointing_gives_same_gradients():
    torch.manual_seed(0)
    a = vim_tiny(num_classes=10, drop_path_rate=0.0, img_size=32, depth=2)
    b = vim_tiny(num_classes=10, drop_path_rate=0.0, img_size=32, depth=2, grad_ckpt=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 32, 32)
    for m in (a, b):
        logits, states = m(x, return_states=True)
        (logits.sum() + sum(s[0].sum() + s[1].sum() for s in states)).backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5), n
