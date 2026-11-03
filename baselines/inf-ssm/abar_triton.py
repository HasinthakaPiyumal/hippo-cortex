"""Fused Triton kernels for A_bar[b, l, n] = mean_d exp(softplus(delta[b, d, l]) * A[d, n]) and its gradients.

The PyTorch version in mamba_vim._MeanDiscreteA loops over n with unfused elementwise ops; on a T4 it
took ~2/3 of an Inf-SSM training step. Here each pass reads delta once and never materialises the
(b, d, l, n) tensor. `available()` runs a one-off comparison against the PyTorch version on the GPU and
returns False (so the caller falls back) if Triton is missing or the results differ.
"""
import threading

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs
    triton = None

_STATE = {'checked': False, 'ok': False}
_LOCK = threading.Lock()  # nn.DataParallel calls the first forward from several threads at once
BLOCK_D, BLOCK_L = 16, 32

if triton is not None:
    @triton.jit
    def _softplus(x):
        # torch.nn.functional.softplus with beta=1, threshold=20
        return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))

    @triton.jit
    def _abar_fwd_kernel(delta_ptr, a_ptr, out_ptr, L, s_db, s_dd, s_dl, s_ad, s_an, s_ob, s_ol, s_on, inv_d,
                         D: tl.constexpr, N: tl.constexpr, BD: tl.constexpr, BL: tl.constexpr):
        b = tl.program_id(0)
        offs_l = tl.program_id(1) * BL + tl.arange(0, BL)
        offs_n = tl.arange(0, N)
        m_l = offs_l < L
        acc = tl.zeros((BL, N), dtype=tl.float32)
        for d0 in range(0, D, BD):
            offs_d = d0 + tl.arange(0, BD)
            m_d = offs_d < D
            dl = tl.load(delta_ptr + b * s_db + offs_d[:, None] * s_dd + offs_l[None, :] * s_dl,
                         mask=m_d[:, None] & m_l[None, :], other=0.0).to(tl.float32)
            a = tl.load(a_ptr + offs_d[:, None] * s_ad + offs_n[None, :] * s_an, mask=m_d[:, None], other=0.0).to(tl.float32)
            e = tl.exp(_softplus(dl)[:, :, None] * a[:, None, :])  # (BD, BL, N)
            e = tl.where(m_d[:, None, None], e, 0.0)
            acc += tl.sum(e, axis=0)
        tl.store(out_ptr + b * s_ob + offs_l[:, None] * s_ol + offs_n[None, :] * s_on, acc * inv_d, mask=m_l[:, None])

    @triton.jit
    def _abar_bwd_kernel(delta_ptr, a_ptr, g_ptr, gdelta_ptr, ga_ptr, L, s_db, s_dd, s_dl, s_ad, s_an, s_gb, s_gl, s_gn,
                         inv_d, D: tl.constexpr, N: tl.constexpr, BD: tl.constexpr, BL: tl.constexpr):
        b = tl.program_id(0)
        offs_d = tl.program_id(1) * BD + tl.arange(0, BD)
        offs_l = tl.program_id(2) * BL + tl.arange(0, BL)
        offs_n = tl.arange(0, N)
        m_d = offs_d < D
        m_l = offs_l < L
        m_dl = m_d[:, None] & m_l[None, :]
        dl = tl.load(delta_ptr + b * s_db + offs_d[:, None] * s_dd + offs_l[None, :] * s_dl, mask=m_dl, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + offs_d[:, None] * s_ad + offs_n[None, :] * s_an, mask=m_d[:, None], other=0.0).to(tl.float32)
        g = tl.load(g_ptr + b * s_gb + offs_l[:, None] * s_gl + offs_n[None, :] * s_gn, mask=m_l[:, None], other=0.0).to(tl.float32)
        sp = _softplus(dl)
        eg = tl.exp(sp[:, :, None] * a[:, None, :]) * g[None, :, :]  # (BD, BL, N) = dOut/dX * grad
        eg = tl.where(m_dl[:, :, None], eg, 0.0)
        # d/d delta: sum_n eg * A * softplus'(delta) / D
        gd = tl.sum(eg * a[:, None, :], axis=2) * tl.sigmoid(dl) * inv_d
        tl.store(gdelta_ptr + b * D * L + offs_d[:, None] * L + offs_l[None, :], gd, mask=m_dl)
        # d/dA: sum_{b, l} eg * softplus(delta) / D, accumulated over the (b, l) blocks
        ga = tl.sum(eg * sp[:, :, None], axis=1) * inv_d
        tl.atomic_add(ga_ptr + offs_d[:, None] * N + offs_n[None, :], ga, mask=m_d[:, None])


def abar_fwd(delta: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    B, D, L = delta.shape
    N = A.shape[1]
    out = torch.empty(B, L, N, device=delta.device, dtype=torch.float32)
    grid = (B, triton.cdiv(L, BLOCK_L))
    _abar_fwd_kernel[grid](delta, A, out, L, *delta.stride(), *A.stride(), *out.stride(), 1.0 / D,
                           D=D, N=N, BD=BLOCK_D, BL=BLOCK_L)
    return out


def abar_bwd(delta: torch.Tensor, A: torch.Tensor, grad: torch.Tensor):
    B, D, L = delta.shape
    N = A.shape[1]
    g_delta = torch.empty(B, D, L, device=delta.device, dtype=torch.float32)
    g_A = torch.zeros(D, N, device=delta.device, dtype=torch.float32)
    grid = (B, triton.cdiv(D, BLOCK_D), triton.cdiv(L, BLOCK_L))
    _abar_bwd_kernel[grid](delta, A, grad, g_delta, g_A, L, *delta.stride(), *A.stride(), *grad.stride(), 1.0 / D,
                           D=D, N=N, BD=BLOCK_D, BL=BLOCK_L)
    return g_delta, g_A


def available(reference_fwd, reference_bwd, device) -> bool:
    """True if the Triton kernels run on `device` and match the PyTorch reference (checked once per process)."""
    if triton is None:
        return False
    with _LOCK:
        if not _STATE['checked']:
            _STATE['checked'] = True
            try:
                gen = torch.Generator(device='cpu').manual_seed(0)
                # non-contiguous delta (b, d, l) with stride(-1) == 1, like the rearranged dt projection in BiMamba
                delta = torch.randn(40, 3, 37, generator=gen).to(device).permute(1, 0, 2)
                A = -(torch.rand(40, 16, generator=gen) * 4 + 0.05).to(device)
                grad = torch.randn(3, 37, 16, generator=gen).to(device)
                ok = torch.allclose(abar_fwd(delta, A), reference_fwd(delta, A), rtol=1e-4, atol=1e-6)
                gd, ga = abar_bwd(delta, A, grad)
                rd, ra = reference_bwd(delta, A, grad)
                ok = ok and torch.allclose(gd, rd, rtol=1e-4, atol=1e-6) and torch.allclose(ga, ra, rtol=1e-4, atol=1e-6)
                _STATE['ok'] = bool(ok)
                print(f":: A_bar Triton kernels {'match the PyTorch reference, using them' if ok else 'DO NOT match the PyTorch reference, falling back'}",
                      flush=True)
            except Exception as e:  # compile/runtime problems: keep training with the PyTorch path
                _STATE['ok'] = False
                print(f":: A_bar Triton kernels unavailable ({type(e).__name__}: {e}), using the PyTorch path", flush=True)
    return _STATE['ok']
