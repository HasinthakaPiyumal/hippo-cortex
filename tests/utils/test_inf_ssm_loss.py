import torch
from baselines.inf_ssm.utils.inf_ssm_loss import InfSSMLoss

def test_sylvester_solver():
    loss_fn = InfSSMLoss()
    
    # Setup random diagonal parameters
    B, L, N = 2, 3, 4
    # A values must be in (-1, 1) for stability
    A1 = torch.rand(B, L, N) * 0.8
    A2 = torch.rand(B, L, N) * 0.8
    C1 = torch.randn(B, L, N)
    C2 = torch.randn(B, L, N)
    
    # Solve G
    G = loss_fn.solve_sylvester_diagonal(A1, C1, A2, C2) # (B, L, N, N)
    
    # Verify the equation: G - A1 G A2^T == C1^T C2
    # For a single element at batch b, sequence l:
    # A1_bl is a diagonal matrix of size N x N
    # A2_bl is a diagonal matrix of size N x N
    # G_bl - A1_bl @ G_bl @ A2_bl == C1_bl^T C2_bl
    for b in range(B):
        for l in range(L):
            g = G[b, l]
            a1 = torch.diag(A1[b, l])
            a2 = torch.diag(A2[b, l])
            c1 = C1[b, l].unsqueeze(-1) # (N, 1)
            c2 = C2[b, l].unsqueeze(0)  # (1, N)
            
            rhs = c1 @ c2
            lhs = g - a1 @ g @ a2
            
            torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-5)


def test_inf_ssm_loss_forward():
    loss_fn = InfSSMLoss()
    B, L, O, N = 2, 3, 5, 4
    
    A_old = torch.randn(B, L, O, N)
    C_old = torch.randn(B, L, N)
    A_new = torch.randn(B, L, O, N, requires_grad=True)
    C_new = torch.randn(B, L, N, requires_grad=True)
    
    loss = loss_fn(A_old, C_old, A_new, C_new)
    
    assert loss.dim() == 0  # Should be a scalar
    assert 0.0 <= loss.item() <= 1.0  # Cosine squared is bounded in [0, 1]
    
    # Verify backpropagation
    loss.backward()
    assert A_new.grad is not None
    assert C_new.grad is not None
