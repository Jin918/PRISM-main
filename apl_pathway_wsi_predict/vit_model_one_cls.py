#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vit_model_one_cls.py

WSI-only survival backbone used when train_flag == 1.

Design goals
------------
- Keep the original input/output contract unchanged.
- Support optional APL bottleneck for prototype-based token compression.
- Remain compatible with train_survival.py and utils_cox.py.

Input
-----
x: torch.Tensor
    Shape (B, 500, 384), i.e. patient-level WSI tokens already extracted upstream.

Output
------
- If APL is OFF:
    pred
- If APL is ON:
    pred, loss_div, loss_bal

where:
    pred      : (B, num_classes)
    loss_div  : scalar tensor reshaped to (1,)
    loss_bal  : scalar tensor reshaped to (1,)
"""

from __future__ import annotations

from functools import partial
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from apl_bottleneck import APLBottleneck, proto_diversity_loss, assignment_balance_loss
except Exception:
    from .apl_bottleneck import APLBottleneck, proto_diversity_loss, assignment_balance_loss


# -----------------------------------------------------------------------------
# Stochastic depth
# -----------------------------------------------------------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """DropPath layer."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------------------------------------------------------
# Input embedding
# -----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """
    Here the input is already a token sequence: (B, N, D).
    This module only applies optional normalization.
    """

    def __init__(self, num_patches: int = 500, embed_dim: int = 768, norm_layer=None) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


# -----------------------------------------------------------------------------
# Transformer blocks
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, ntok, dim = x.shape

        qkv = self.qkv(x).reshape(
            bsz, ntok, 3, self.num_heads, dim // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, ntok, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
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
        qk_scale: Optional[float] = None,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
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
            act_layer=act_layer,
            drop=drop_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------------------------------------------------------
# WSI-only Vision Transformer
# -----------------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(
        self,
        wsi_patches: int = 500,
        num_classes: int = 1,
        wsi_dim: int = 384,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        representation_size: Optional[int] = None,
        distilled: bool = False,
        drop_ratio: float = 0.0,
        attn_drop_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        proto_k: int = 0,
        proto_tau: float = 0.07,
        use_apl: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.embed_dim = int(wsi_dim)
        self.num_features = self.embed_dim
        self.num_tokens = 2 if distilled else 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(num_patches=wsi_patches, embed_dim=wsi_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, wsi_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, wsi_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, wsi_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # APL
        self.proto_k = int(proto_k) if proto_k is not None else 0
        if use_apl is None:
            self.use_apl = self.proto_k > 0
        else:
            self.use_apl = bool(use_apl)

        self.proto_tau = float(proto_tau)
        self.apl_wsi = None
        if self.use_apl and self.proto_k > 0:
            self.apl_wsi = APLBottleneck(dim=wsi_dim, K=self.proto_k, tau=self.proto_tau)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=wsi_dim,
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
        ])
        self.norm = norm_layer(wsi_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = int(representation_size)
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(wsi_dim, representation_size)),
                ("act", nn.Tanh()),
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Heads
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        self.apply(_init_vit_weights)

    def forward_features(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]]:
        """
        Returns:
        - without APL: feat
        - with APL   : feat, loss_div, loss_bal
        """
        x = self.patch_embed(x)  # (B, N, D)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # Intentionally do not add pos_embed here, to preserve original behavior and
        # avoid token-length mismatch after optional APL compression.
        # x = self.pos_drop(x + self.pos_embed)

        loss_div = None
        loss_bal = None

        if self.apl_wsi is not None:
            prefix = x[:, :self.num_tokens, :]
            patch_tokens = x[:, self.num_tokens:, :]

            proto_tokens, assign = self.apl_wsi(patch_tokens)
            x = torch.cat((prefix, proto_tokens), dim=1)

            loss_div = proto_diversity_loss(self.apl_wsi.prototypes)
            loss_bal = assignment_balance_loss(assign)

        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            feat = self.pre_logits(x[:, 0])
        else:
            feat = (x[:, 0], x[:, 1])

        if self.apl_wsi is not None:
            return feat, loss_div, loss_bal
        return feat

    def forward(self, x: torch.Tensor):
        out = self.forward_features(x)

        if self.apl_wsi is not None:
            feat, loss_div, loss_bal = out
        else:
            feat = out
            loss_div, loss_bal = None, None

        if self.head_dist is not None:
            pred_cls = self.head(feat[0])
            pred_dist = self.head_dist(feat[1])
            pred = (pred_cls + pred_dist) / 2.0
        else:
            pred = self.head(feat)

        if self.apl_wsi is None:
            return pred

        loss_div = torch.as_tensor(loss_div, device=pred.device, dtype=pred.dtype)
        loss_bal = torch.as_tensor(loss_bal, device=pred.device, dtype=pred.dtype)

        if loss_div.numel() > 1:
            loss_div = loss_div.mean()
        if loss_bal.numel() > 1:
            loss_bal = loss_bal.mean()

        return pred, loss_div.reshape(1), loss_bal.reshape(1)


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def _init_vit_weights(m: nn.Module) -> None:
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
    has_logits: bool = True,
    wsi_block: int = 12,
    dpr: float = 0.1,
    proto_k: int = 0,
    proto_tau: float = 0.07,
    use_apl: Optional[bool] = None,
    **kwargs,
):
    """
    Public constructor used by train_survival.py.

    Parameters kept compatible with your existing training code.
    """
    model = VisionTransformer(
        wsi_patches=500,
        wsi_dim=384,
        depth=wsi_block,
        num_heads=16,
        representation_size=384 if has_logits else None,
        drop_path_ratio=dpr,
        drop_ratio=dpr,
        attn_drop_ratio=dpr,
        num_classes=num_classes,
        proto_k=proto_k,
        proto_tau=proto_tau,
        use_apl=use_apl,
    )
    return model
