import torch
import torch.nn as nn

class InfSSMLoss(nn.Module):
    """
    Inf-SSM (Lee et al., 2026) Observability Subspace Regularization Loss.
    Computes Grassmannian distance between old and new model states.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def soft_normalize(self, x):
        """Soft-normalization function to enforce stability (SN(x) = 2 / (1 + exp(-x)) - 1)."""
        return 2.0 / (1.0 + torch.exp(-x)) - 1.0

    def solve_sylvester_diagonal(self, A1, C1, A2, C2):
        """
        Solves the discrete Sylvester equation G - A1 G A2^T = C1^T C2
        for diagonal A1, A2 in O(N^2) using outer products and broadcasting.
        A1, A2: (B, L, N)
        C1, C2: (B, L, N)
        Returns: G of shape (B, L, N, N)
        """
        # Outer product of C1 and C2 -> shape (B, L, N, N)
        numerator = C1.unsqueeze(-1) * C2.unsqueeze(-2)
        # Outer product of A1 and A2 -> shape (B, L, N, N)
        denom = 1.0 - (A1.unsqueeze(-1) * A2.unsqueeze(-2))
        
        G = numerator / (denom + self.eps)
        return G

    def forward(self, A_old, C_old, A_new, C_new):
        """
        Args:
            A_old: (B, L, O, N) or (B, L, N) - Old model state transitions
            C_old: (B, L, N) - Old model output mapping
            A_new: (B, L, O, N) or (B, L, N) - New model state transitions
            C_new: (B, L, N) - New model output mapping
        """
        # Average A across the channel/outer dimension 'O' if it has 4 dimensions
        if A_old.dim() == 4:
            A_old = A_old.mean(dim=2)
        if A_new.dim() == 4:
            A_new = A_new.mean(dim=2)

        # Apply soft-normalization
        A_old_tilde = self.soft_normalize(A_old) # (B, L, N)
        C_old_tilde = self.soft_normalize(C_old) # (B, L, N)
        A_new_tilde = self.soft_normalize(A_new) # (B, L, N)
        C_new_tilde = self.soft_normalize(C_new) # (B, L, N)

        # Solve for the Gram matrices (G1, G2, G3)
        G1 = self.solve_sylvester_diagonal(A_old_tilde, C_old_tilde, A_old_tilde, C_old_tilde)
        G2 = self.solve_sylvester_diagonal(A_new_tilde, C_new_tilde, A_new_tilde, C_new_tilde)
        G3 = self.solve_sylvester_diagonal(A_old_tilde, C_old_tilde, A_new_tilde, C_new_tilde)

        # Traces (sum of diagonals) -> shape (B, L)
        tr_G1 = torch.diagonal(G1, dim1=-2, dim2=-1).sum(dim=-1)
        tr_G2 = torch.diagonal(G2, dim1=-2, dim2=-1).sum(dim=-1)

        # Frobenius Norm squared of G3 -> shape (B, L)
        norm_G3_sq = (G3 ** 2).sum(dim=(-2, -1))

        # Squared cosine of the principal angle
        cos_sq = norm_G3_sq / (tr_G1 * tr_G2 + self.eps)
        
        # Clamp to [0, 1] for numerical stability
        cos_sq = torch.clamp(cos_sq, min=0.0, max=1.0)

        # Inf-SSM Loss: 1 - cos^2(theta)
        loss = 1.0 - cos_sq.mean()
        return loss
