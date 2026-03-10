#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vit_model_gene_wsi_concat_predict.py

Prediction-time multimodal survival model with optional attention export.

This module is architecturally aligned with the training-time multimodal
survival model. The only functional difference is that the prediction model
can optionally return cross-attention maps and APL assignment matrices for
downstream interpretability analysis.

Forward output:
- return_attn=False:
    (gene2wsi_feature, pred_head, loss_div, loss_bal)
- return_attn=True:
    (gene2wsi_feature, pred_head, loss_div, loss_bal, attn, assign)

Notes
-----
- When APL is enabled (proto_k > 0), cross-attention has shape (B, H, G, K),
  where K is the number of prototypes rather than the original number of WSI
  tokens. The returned `assign` tensor has shape (B, P, K) and can be used to
  map prototype-level attention back to patch-level space.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn

from apl_bottleneck import (
    APLBottleneck,
    assignment_balance_loss,
    proto_diversity_loss,
)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth applied per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Stochastic depth layer."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    Input x is already a sequence of patch embeddings with shape (B, L, C).
    This module only applies normalization.
    """

    def __init__(self, embed_dim: int = 768, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class EmbedReduction(nn.Module):
    """Token-wise MLP projection: (B, L, Cin) -> (B, L, Cout)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = 1280,
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
    """Multi-head self-attention."""

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
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GeneGuidedTransformerFusion(nn.Module):
    """
    Cross-attention module in which gene tokens query WSI tokens.

    Inputs
    ------
    x_wsi : (B, M, C)
        WSI token sequence.
    x_gene : (B, N, C)
        Gene token sequence.

    Returns
    -------
    x : (B, N, C)
        Cross-attended gene features.
    attn : (B, H, N, M)
        Cross-attention weights.
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
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x_wsi: torch.Tensor, x_gene: torch.Tensor):
        if x_wsi.shape[-1] != x_gene.shape[-1]:
            raise ValueError("WSI and gene token dimensions must match.")

        batch_size, num_wsi, dim = x_wsi.shape
        _, num_gene, _ = x_gene.shape

        q = self.q(x_gene).reshape(
            batch_size, num_gene, self.num_heads, dim // self.num_heads
        ).permute(0, 2, 1, 3)
        kv = self.kv(x_wsi).reshape(
            batch_size, num_wsi, 2, self.num_heads, dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_gene, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    """Transformer MLP block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
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


class Block(nn.Module):
    """Standard Transformer encoder block."""

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


class VisionTransformer(nn.Module):
    """Multimodal survival transformer with optional APL and attention export."""

    def __init__(
        self,
        wsi_patches: int = 500,
        gene_patches: int = 236,
        embed_wsi_dim: int = 384,
        embed_gene_dim: int = 256,
        num_classes: int = 1,
        proto_k: int = 4,
        proto_tau: float = 0.1,
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
        return_attn: bool = False,
    ):
        super().__init__()

        self.return_attn = bool(return_attn)

        self.wsi_patches = int(wsi_patches)
        self.gene_patches = int(gene_patches)
        self.embed_wsi_dim = int(embed_wsi_dim)
        self.embed_gene_dim = int(embed_gene_dim)
        self.num_classes = int(num_classes)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.patch_embed = embed_layer(embed_dim=self.embed_wsi_dim)
        self.gene_embed = EmbedReduction(
            in_features=6292,
            hidden_features=1000,
            out_features=self.embed_gene_dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )

        self.pos_wsi_embed = nn.Parameter(torch.zeros(1, self.wsi_patches, self.embed_wsi_dim))
        self.pos_gene_embed = nn.Parameter(torch.zeros(1, self.gene_patches, self.embed_gene_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        nn.init.trunc_normal_(self.pos_wsi_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_gene_embed, std=0.02)

        self.proto_k = int(proto_k) if proto_k is not None else 0
        self.use_apl = self.proto_k > 0

        if self.use_apl:
            self.apl_wsi = APLBottleneck(dim=self.embed_wsi_dim, K=self.proto_k, tau=float(proto_tau))
            self.pos_wsi_proto_embed = nn.Parameter(torch.zeros(1, self.proto_k, self.embed_wsi_dim))
            nn.init.trunc_normal_(self.pos_wsi_proto_embed, std=0.02)
        else:
            self.apl_wsi = None
            self.pos_wsi_proto_embed = None

        dpr_wsi = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_wsi)]
        self.blocks_wsi = nn.Sequential(
            *[
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
            ]
        )

        dpr_gene = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_gene)]
        self.blocks_gene = nn.Sequential(
            *[
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
            ]
        )

        self.norm_wsi = norm_layer(self.embed_wsi_dim)
        self.norm_gene = norm_layer(self.embed_gene_dim)

        self.wsi_embed_reduction = EmbedReduction(
            in_features=self.embed_wsi_dim,
            hidden_features=640,
            out_features=self.embed_gene_dim,
            act_layer=act_layer,
            drop=drop_ratio,
        )

        self.gene_guided_wsi_fusion = GeneGuidedTransformerFusion(
            dim=self.embed_gene_dim,
            num_heads=num_heads,
            q_bias=False,
            kv_bias=False,
            qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
        )

        self.head = (
            nn.Linear(self.gene_patches * 2, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

        self.apply(_init_vit_weights)

    def forward_features_wsi(self, x: torch.Tensor):
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
        x = self.pos_drop(x + self.pos_gene_embed)
        x = self.blocks_gene(x)
        x = self.norm_gene(x)
        return x

    def forward(self, x_wsi: torch.Tensor, x_gene: torch.Tensor):
        wsi_features, assign = self.forward_features_wsi(x_wsi)
        wsi_features_reduction = self.wsi_embed_reduction(wsi_features)

        if self.apl_wsi is not None and wsi_features_reduction.size(1) != self.proto_k:
            raise RuntimeError(
                f"APL token mismatch: got {wsi_features_reduction.size(1)} vs proto_k={self.proto_k}"
            )

        gene_features_reduction = self.gene_embed(x_gene)
        gene_features = self.forward_features_gene(gene_features_reduction)

        logit_scale = self.logit_scale.exp()
        gene2wsi_feature = logit_scale * (
            gene_features_reduction @ wsi_features_reduction.transpose(-2, -1)
        )

        x_fusion, attn = self.gene_guided_wsi_fusion(
            wsi_features_reduction, gene_features
        )

        x_fusion_gap = x_fusion.mean(dim=-1, keepdim=True)
        gene_gap = gene_features.mean(dim=-1, keepdim=True)

        fused_features = torch.cat([gene_gap, x_fusion_gap], dim=1).squeeze(-1)
        pred_head = self.head(fused_features)

        if self.apl_wsi is not None:
            loss_div = proto_diversity_loss(self.apl_wsi.prototypes)
            loss_bal = assignment_balance_loss(assign)
            loss_div = torch.as_tensor(loss_div, device=pred_head.device).reshape(1)
            loss_bal = torch.as_tensor(loss_bal, device=pred_head.device).reshape(1)
        else:
            loss_div = torch.zeros(1, device=pred_head.device)
            loss_bal = torch.zeros(1, device=pred_head.device)

        if self.return_attn:
            return gene2wsi_feature, pred_head, loss_div, loss_bal, attn, assign
        return gene2wsi_feature, pred_head, loss_div, loss_bal


def _init_vit_weights(module: nn.Module) -> None:
    """Vision Transformer parameter initialization."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def my_model(
    num_classes: int = 1,
    has_logits: bool = True,   # kept for interface compatibility
    wsi_block: int = 12,
    gene_block: int = 3,
    dpr: float = 0.1,
    proto_k: int = 4,
    proto_tau: float = 0.1,
    return_attn: bool = False,
):
    _ = has_logits  # retained for compatibility with the training script interface

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
        return_attn=return_attn,
    )
    return model
