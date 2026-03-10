#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gene_only.py

Gene-only survival model for PRISM / PAMT-style pathway-token modeling.

Design goal
-----------
This module is the official gene-only branch used when `train_flag == 2`.

Input
-----
x_gene: torch.Tensor
    Shape: [B, gene_patches, in_features]
    Default expected shape: [B, 236, 6292]

Output
------
pred: torch.Tensor
    Shape: [B, num_classes]
    Default: [B, 1], representing Cox risk score / log-risk.

Compatibility
-------------
Public factory:
    my_gene_only_model(...)

This interface is kept compatible with the current training pipeline.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Stochastic Depth
# -----------------------------------------------------------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (stochastic depth) per sample.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    drop_prob : float
        Probability of dropping the residual branch.
    training : bool
        Whether the module is in training mode.

    Returns
    -------
    torch.Tensor
        Output tensor after stochastic depth.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """nn.Module wrapper for stochastic depth."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob) if drop_prob is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------------------------------------------------------
# Token embedding
# -----------------------------------------------------------------------------
class EmbedReduction(nn.Module):
    """
    Map per-pathway gene vector to token embedding.

    Input:
        x: [B, P, in_features]
    Output:
        y: [B, P, out_features]
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: int = 256,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.norm1 = norm_layer(hidden_features)
        self.drop = nn.Dropout(drop)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm2 = norm_layer(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm1(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)
        return x


# -----------------------------------------------------------------------------
# Transformer blocks
# -----------------------------------------------------------------------------
class Mlp(nn.Module):
    """Transformer MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionGENE(nn.Module):
    """
    Multi-head self-attention for gene/pathway tokens.

    Note
    ----
    This implementation intentionally preserves the original behavior of your
    previous code, including the internal normalization, to avoid changing the
    effective architecture used by the training pipeline.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: [B, P, C]

        Returns
        -------
        torch.Tensor
            Shape: [B, P, C]
        """
        bsz, num_tokens, dim = x.shape

        x = self.norm(x)
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, P, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class BlockGENE(nn.Module):
    """Standard transformer encoder block for gene/pathway tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = AttentionGENE(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def _init_vit_weights(module: nn.Module) -> None:
    """ViT-style weight initialization."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# -----------------------------------------------------------------------------
# Gene-only survival model
# -----------------------------------------------------------------------------
class GeneOnlyTransformer(nn.Module):
    """
    Gene-only survival backbone.

    Pipeline
    --------
    x_gene [B, P, G]
      -> pathway token embedding
      -> + positional embedding
      -> transformer encoder
      -> token-wise mean over channel dimension
      -> linear Cox head

    This preserves the original output behavior:
        pred: [B, num_classes]
    """

    def __init__(
        self,
        gene_patches: int = 236,
        in_features: int = 6292,
        embed_dim: int = 256,
        depth: int = 1,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_ratio: float = 0.1,
        attn_drop_ratio: float = 0.1,
        drop_path_ratio: float = 0.1,
        norm_layer: Optional[type[nn.Module]] = None,
        act_layer: Optional[type[nn.Module]] = None,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.gene_patches = int(gene_patches)
        self.in_features = int(in_features)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)

        self.gene_embed = EmbedReduction(
            in_features=self.in_features,
            hidden_features=self.embed_dim,
            out_features=self.embed_dim,
            act_layer=act_layer,
            drop=drop_ratio,
            norm_layer=nn.LayerNorm,
        )

        self.pos_gene_embed = nn.Parameter(torch.zeros(1, self.gene_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks_gene = nn.Sequential(
            *[
                BlockGENE(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    drop_path_ratio=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm_gene = norm_layer(self.embed_dim)

        # Preserve original head behavior:
        # [B, P, C] -> mean over C -> [B, P] -> Linear(P -> num_classes)
        self.head = nn.Linear(self.gene_patches, self.num_classes) if self.num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_gene_embed, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x_gene: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_gene : torch.Tensor
            Shape: [B, gene_patches, in_features]

        Returns
        -------
        torch.Tensor
            Shape: [B, num_classes]
        """
        x = self.gene_embed(x_gene)                 # [B, P, 256]
        x = self.pos_drop(x + self.pos_gene_embed)  # [B, P, 256]
        x = self.blocks_gene(x)
        x = self.norm_gene(x)                       # [B, P, 256]

        token_scalar = x.mean(dim=-1)               # [B, P]
        pred = self.head(token_scalar)              # [B, num_classes]
        return pred


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------
def my_gene_only_model(
    num_classes: int = 1,
    gene_block: int = 1,
    dpr: float = 0.1,
    gene_patches: int = 236,
    in_features: int = 6292,
    embed_dim: int = 256,
    num_heads: int = 16,
) -> GeneOnlyTransformer:
    """
    Public constructor used by the training pipeline.

    Parameters
    ----------
    num_classes : int
        Output dimension. For Cox survival training, use 1.
    gene_block : int
        Number of transformer encoder blocks.
    dpr : float
        Shared dropout / attention dropout / drop-path ratio.
    gene_patches : int
        Number of pathway tokens.
    in_features : int
        Per-token input feature dimension.
    embed_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.

    Returns
    -------
    GeneOnlyTransformer
    """
    return GeneOnlyTransformer(
        gene_patches=gene_patches,
        in_features=in_features,
        embed_dim=embed_dim,
        depth=gene_block,
        num_heads=num_heads,
        drop_ratio=dpr,
        attn_drop_ratio=dpr,
        drop_path_ratio=dpr,
        num_classes=num_classes,
    )
