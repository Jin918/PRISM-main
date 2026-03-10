# gene_wsi_predict/apl_bottleneck.py

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class APLBottleneck(nn.Module):
    """
    Attention-based prototype learning (APL) bottleneck.

    This module compresses patch-level tokens into a fixed number of
    prototype tokens.

    Input
    -----
    x : torch.Tensor
        Shape (B, N, D), where
        - B: batch size
        - N: number of patch tokens
        - D: token embedding dimension

    Output
    ------
    proto : torch.Tensor
        Shape (B, K, D), aggregated prototype tokens.

    assign : torch.Tensor
        Shape (B, N, K), soft assignment weights from each patch token
        to each prototype.

    Notes
    -----
    - The learnable prototype codebook has shape (K, D).
    - Assignment is computed by projected token-prototype similarity.
    - Prototype tokens are obtained by weighted averaging of input tokens.
    """

    def __init__(
        self,
        dim: int,
        K: int = 8,
        tau: float = 0.07,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(f"`dim` must be positive, got {dim}.")
        if K <= 0:
            raise ValueError(f"`K` must be positive, got {K}.")
        if tau <= 0:
            raise ValueError(f"`tau` must be positive, got {tau}.")
        if eps <= 0:
            raise ValueError(f"`eps` must be positive, got {eps}.")

        self.dim = dim
        self.K = K
        self.tau = tau
        self.eps = eps

        # Learnable prototype codebook: (K, D)
        self.prototypes = nn.Parameter(torch.empty(K, dim))

        # Linear projections for token-prototype assignment
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters.

        We use Xavier initialization for projection layers and a small-normal
        initialization for the prototype codebook.
        """
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"K={self.K}, "
            f"tau={self.tau}, "
            f"eps={self.eps}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input patch tokens of shape (B, N, D).

        Returns
        -------
        proto : torch.Tensor
            Prototype tokens of shape (B, K, D).

        assign : torch.Tensor
            Assignment matrix of shape (B, N, K).
        """
        if x.ndim != 3:
            raise ValueError(
                f"`x` must have shape (B, N, D), but got shape {tuple(x.shape)}."
            )

        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(
                f"Last dimension of x must equal dim={self.dim}, but got {D}."
            )

        # Project tokens and prototypes into assignment space
        q = self.q(x)                   # (B, N, D)
        k = self.k(self.prototypes)     # (K, D)

        # Similarity logits -> soft assignment
        logits = torch.einsum("bnd,kd->bnk", q, k) / self.tau   # (B, N, K)
        assign = F.softmax(logits, dim=-1)                      # (B, N, K)

        # Weighted pooling from patch tokens to prototype tokens
        proto = torch.einsum("bnk,bnd->bkd", assign, x)         # (B, K, D)
        denom = assign.sum(dim=1, keepdim=False).unsqueeze(-1)  # (B, K, 1)
        proto = proto / (denom + self.eps)

        return proto, assign


def proto_diversity_loss(P: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage learned prototypes to be diverse rather than collapsing
    to highly similar directions.

    Parameters
    ----------
    P : torch.Tensor
        Prototype matrix of shape (K, D).
    eps : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Scalar diversity regularization loss.

    Notes
    -----
    This computes the squared off-diagonal energy of the cosine-similarity
    Gram matrix after L2 normalization.
    """
    if P.ndim != 2:
        raise ValueError(f"`P` must have shape (K, D), but got {tuple(P.shape)}.")

    K = P.size(0)
    if K <= 1:
        # No pairwise diversity can be defined for K=1
        return P.new_zeros(())

    Pn = F.normalize(P, p=2, dim=-1, eps=eps)          # (K, D)
    gram = Pn @ Pn.t()                                 # (K, K)

    eye = torch.eye(K, device=gram.device, dtype=gram.dtype)
    off_diag = gram - eye

    return (off_diag ** 2).sum() / (K * (K - 1) + eps)


def assignment_balance_loss(assign: torch.Tensor) -> torch.Tensor:
    """
    Encourage balanced prototype usage across the mini-batch.

    Parameters
    ----------
    assign : torch.Tensor
        Soft assignment matrix of shape (B, N, K).

    Returns
    -------
    torch.Tensor
        Scalar balance regularization loss.

    Notes
    -----
    The target usage is uniform over K prototypes.
    """
    if assign.ndim != 3:
        raise ValueError(
            f"`assign` must have shape (B, N, K), but got {tuple(assign.shape)}."
        )

    K = assign.size(-1)
    if K <= 0:
        raise ValueError(f"Invalid prototype dimension K={K}.")

    usage = assign.mean(dim=(0, 1))                    # (K,)
    target = torch.full_like(usage, fill_value=1.0 / K)

    return ((usage - target) ** 2).mean()
