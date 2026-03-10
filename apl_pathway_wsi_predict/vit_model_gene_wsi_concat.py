#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vit_model_gene_wsi_concat.py

Multimodal survival model for WSI + pathway/gene-program inputs.

Design principles
-----------------
1. Preserve the original architecture and default behavior used in the study.
2. Add explicit switches for:
      - return_contrastive: whether to return the contrastive similarity tensor
      - use_apl: whether to enable the prototype bottleneck
3. Keep outputs stable and explicit for downstream training/evaluation code.
4. Use fail-fast shape checks to avoid silent bugs.

Default behavior
----------------
By default, this file preserves the original main-model behavior:
    return (gene2wsi_feature, pred_head, loss_div, loss_bal)

If return_contrastive=False:
    return (pred_head, None, loss_div, loss_bal)

This makes strict no-contrastive ablation possible without altering the
underlying architecture.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from apl_bottleneck import APLBottleneck, proto_diversity_loss, assignment_balance_loss


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    Input x is already patch embeddings with shape (B, L, C).
    This module only applies normalization.
    """
    def __init__(self, embed_dim: int = 768, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class EmbedReduction(nn.Module):
    """
    Token-wise MLP:
        (B, L, in_features) -> (B, L, out_features)
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = 1280,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Gene_Guided_Transformer_Fusion(nn.Module):
    """
    Cross-attention:
      gene queries attend to WSI keys/values.

    Inputs
    ------
    x1 : WSI tokens, shape (B, M, C)
    x2 : Gene tokens, shape (B, N, C)

    Returns
    -------
    out : shape (B, N, C)
    """
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 16,
        q_bias: bool = False,
        kv_bias: bool = False,
        qk_scale=None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.shape[-1] != x2.shape[-1]:
            raise ValueError(
                f"Fusion dim mismatch: x1 last dim={x1.shape[-1]} vs x2 last dim={x2.shape[-1]}"
            )

        B1, M, C1 = x1.shape
        B2, N, C2 = x2.shape
        if B1 != B2 or C1 != C2:
            raise ValueError(
                f"Fusion shape mismatch: x1={tuple(x1.shape)}, x2={tuple(x2.shape)}"
            )

        B, C = B1, C1

        q = self.q(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x1).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
        )
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(
        self,
        wsi_patches: int = 500,
        gene_patches: int = 236,
        embed_wsi_dim: int = 384,
        embed_gene_dim: int = 256,
        num_classes: int = 1,
        proto_k: int = 0,
        proto_tau: float = 0.07,
        use_apl: Optional[bool] = None,
        return_contrastive: bool = True,
        depth_gene: int = 3,
        depth_wsi: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()

        self.wsi_patches = int(wsi_patches)
        self.gene_patches = int(gene_patches)
        self.embed_wsi_dim = int(embed_wsi_dim)
        self.embed_gene_dim = int(embed_gene_dim)
        self.num_classes = int(num_classes)
        self.proto_k = int(proto_k) if proto_k is not None else 0
        self.proto_tau = float(proto_tau)
        self.return_contrastive = bool(return_contrastive)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Explicit APL switch
        if use_apl is None:
            self.use_apl = self.proto_k > 0
        else:
            self.use_apl = bool(use_apl)
            if not self.use_apl:
                self.proto_k = 0

        # Contrastive temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Input embedding
        self.patch_embed = embed_layer(embed_dim=self.embed_wsi_dim)
        self.gene_embed = EmbedReduction(
            in_features=6292,
            hidden_features=1000,
            out_features=self.embed_gene_dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )

        # Positional embeddings
        self.pos_wsi_embed = nn.Parameter(torch.zeros(1, self.wsi_patches, self.embed_wsi_dim))
        self.pos_gene_embed = nn.Parameter(torch.zeros(1, self.gene_patches, self.embed_gene_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        nn.init.trunc_normal_(self.pos_wsi_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_gene_embed, std=0.02)

        # APL bottleneck
        if self.use_apl:
            if self.proto_k <= 0:
                raise ValueError("use_apl=True requires proto_k > 0")
            self.apl_wsi = APLBottleneck(dim=self.embed_wsi_dim, K=self.proto_k, tau=self.proto_tau)
            self.pos_wsi_proto_embed = nn.Parameter(torch.zeros(1, self.proto_k, self.embed_wsi_dim))
            nn.init.trunc_normal_(self.pos_wsi_proto_embed, std=0.02)
        else:
            self.apl_wsi = None
            self.pos_wsi_proto_embed = None

        # WSI transformer
        dpr_wsi = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_wsi)]
        self.blocks_wsi = nn.Sequential(*[
            Block(
                dim=self.embed_wsi_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=dpr_wsi[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth_wsi)
        ])

        # Gene transformer
        dpr_gene = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_gene)]
        self.blocks_gene = nn.Sequential(*[
            Block(
                dim=self.embed_gene_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=dpr_gene[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth_gene)
        ])

        self.norm_wsi = norm_layer(self.embed_wsi_dim)
        self.norm_gene = norm_layer(self.embed_gene_dim)

        # Reduce WSI embedding dim to gene dim
        self.wsi_embed_reduction = EmbedReduction(
            in_features=self.embed_wsi_dim,
            hidden_features=640,
            out_features=self.embed_gene_dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )

        # Cross-modal fusion
        self.gene_guided_wsi_fusion = Gene_Guided_Transformer_Fusion(dim=self.embed_gene_dim)

        # Head
        # Keep original design:
        # concat(gene_gap, fusion_gap) => (B, 236*2)
        self.head = nn.Linear(self.gene_patches * 2, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(_init_vit_weights)

    # -------------------------------------------------------------------------
    # Branch forward
    # -------------------------------------------------------------------------
    def forward_features_wsi(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, 500, 384), already extracted WSI patch features

        Returns
        -------
        wsi_tokens:
            (B, K, 384) if APL is enabled
            (B, 500, 384) otherwise
        assign:
            (B, 500, K) if APL is enabled
            None otherwise
        """
        if x.ndim != 3:
            raise ValueError(f"WSI input must be 3D (B,L,C), got shape={tuple(x.shape)}")
        if x.shape[1] != self.wsi_patches or x.shape[2] != self.embed_wsi_dim:
            raise ValueError(
                f"WSI input shape mismatch: expected (*,{self.wsi_patches},{self.embed_wsi_dim}), "
                f"got {tuple(x.shape)}"
            )

        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_wsi_embed)

        if self.apl_wsi is None:
            x = self.blocks_wsi(x)
            x = self.norm_wsi(x)
            return x, None

        x_proto, assign = self.apl_wsi(x)
        x_proto = self.pos_drop(x_proto + self.pos_wsi_proto_embed)
        x_proto = self.blocks_wsi(x_proto)
        x_proto = self.norm_wsi(x_proto)
        return x_proto, assign

    def forward_features_gene(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Gene token input must be 3D (B,L,C), got shape={tuple(x.shape)}")
        if x.shape[1] != self.gene_patches or x.shape[2] != self.embed_gene_dim:
            raise ValueError(
                f"Gene token shape mismatch: expected (*,{self.gene_patches},{self.embed_gene_dim}), "
                f"got {tuple(x.shape)}"
            )
        x = self.pos_drop(x + self.pos_gene_embed)
        x = self.blocks_gene(x)
        x = self.norm_gene(x)
        return x

    # -------------------------------------------------------------------------
    # Main forward
    # -------------------------------------------------------------------------
    def forward(self, x_wsi: torch.Tensor, x_gene: torch.Tensor):
        """
        Parameters
        ----------
        x_wsi : (B, 500, 384)
        x_gene: (B, 236, 6292)

        Returns
        -------
        if return_contrastive:
            (gene2wsi_feature, pred_head, loss_div, loss_bal)
        else:
            (pred_head, None, loss_div, loss_bal)
        """
        if x_gene.ndim != 3:
            raise ValueError(f"Gene input must be 3D (B,L,G), got shape={tuple(x_gene.shape)}")
        if x_gene.shape[1] != self.gene_patches:
            raise ValueError(
                f"Gene input length mismatch: expected {self.gene_patches}, got {x_gene.shape[1]}"
            )

        # WSI branch
        wsi_tokens, assign = self.forward_features_wsi(x_wsi)
        wsi_tokens_red = self.wsi_embed_reduction(wsi_tokens)

        # Gene branch
        gene_tokens_red = self.gene_embed(x_gene)
        gene_tokens = self.forward_features_gene(gene_tokens_red)

        # Contrastive similarity: (B, 236, M_wsi)
        logit_scale = self.logit_scale.exp()
        gene2wsi_feature = logit_scale * (gene_tokens_red @ wsi_tokens_red.transpose(-2, -1))

        # Cross-modal fusion
        x_fusion = self.gene_guided_wsi_fusion(wsi_tokens_red, gene_tokens)

        # Head input
        x_fusion_gap = x_fusion.mean(dim=-1, keepdim=True)   # (B,236,1)
        gene_gap = gene_tokens.mean(dim=-1, keepdim=True)    # (B,236,1)
        fused = torch.cat([gene_gap, x_fusion_gap], dim=1).squeeze(-1)  # (B,472)

        pred_head = self.head(fused)  # (B,1)

        # APL regularization
        if self.apl_wsi is not None:
            loss_div = proto_diversity_loss(self.apl_wsi.prototypes)
            loss_bal = assignment_balance_loss(assign)
            loss_div = torch.as_tensor(loss_div, device=pred_head.device).reshape(1)
            loss_bal = torch.as_tensor(loss_bal, device=pred_head.device).reshape(1)
        else:
            loss_div = torch.zeros(1, device=pred_head.device)
            loss_bal = torch.zeros(1, device=pred_head.device)

        if self.return_contrastive:
            return gene2wsi_feature, pred_head, loss_div, loss_bal

        return pred_head, None, loss_div, loss_bal


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------
def my_model(
    num_classes: int = 1,
    has_logits: bool = True,   # kept only for compatibility
    wsi_block: int = 12,
    gene_block: int = 3,
    dpr: float = 0.1,
    proto_k: int = 0,
    proto_tau: float = 0.07,
    use_apl: Optional[bool] = None,
    return_contrastive: bool = True,
    **kwargs,
):
    """
    Public constructor used by train_survival.py.

    Default behavior remains aligned with the original main model.
    """
    model = VisionTransformer(
        wsi_patches=500,
        gene_patches=236,
        embed_wsi_dim=384,
        embed_gene_dim=256,
        depth_gene=gene_block,
        depth_wsi=wsi_block,
        num_heads=16,
        drop_path_ratio=dpr,
        drop_ratio=dpr,
        attn_drop_ratio=dpr,
        num_classes=num_classes,
        proto_k=proto_k,
        proto_tau=proto_tau,
        use_apl=use_apl,
        return_contrastive=return_contrastive,
    )
    return model
