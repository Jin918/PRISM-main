#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vit_model_gene_wsi_concat_no_contrastive_loss.py

Strict ablation entry for the "w/o contrastive loss" model.

Design goals
------------
1. Reuse the same multimodal backbone definition from
   `vit_model_gene_wsi_concat.py` to keep the architecture aligned with the
   main model as closely as possible.
2. Explicitly disable the contrastive branch/output at model construction time.
3. If the base model does not expose a valid contrastive-off switch, raise an
   error instead of silently faking a no-contrastive ablation.
4. Normalize forward outputs into a stable format:
      (pred, attn, loss_div, loss_bal)
   so downstream training/evaluation code can remain unchanged.

This is a publication-oriented implementation: strict, explicit, and fail-fast.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Robust import
# -----------------------------------------------------------------------------
try:
    from vit_model_gene_wsi_concat import my_model as _concat_my_model
except Exception:
    from .vit_model_gene_wsi_concat import my_model as _concat_my_model


# -----------------------------------------------------------------------------
# Introspection helpers
# -----------------------------------------------------------------------------
def _get_signature(fn) -> inspect.Signature:
    try:
        return inspect.signature(fn)
    except Exception as exc:
        raise RuntimeError(f"Cannot inspect signature of {_concat_my_model}: {exc}") from exc


def _supports_kw(fn, kw: str) -> bool:
    sig = _get_signature(fn)
    if kw in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _filter_supported_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs explicitly supported by the target callable unless it
    accepts **kwargs, in which case everything is passed through.
    """
    sig = _get_signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _choose_contrastive_off_kw(fn) -> Tuple[str, Any]:
    """
    Select a strict and explicit contrastive-off keyword supported by the base
    model constructor. If no such keyword exists, raise an error.

    Supported naming conventions:
      - return_contrastive=False
      - contrastive=False
      - use_contrastive=False
      - contrastive_loss_flag=0
    """
    candidates = [
        ("return_contrastive", False),
        ("contrastive", False),
        ("use_contrastive", False),
        ("contrastive_loss_flag", 0),
    ]
    for name, value in candidates:
        if _supports_kw(fn, name):
            return name, value

    raise RuntimeError(
        "The base multimodal model does not expose an explicit switch to disable "
        "the contrastive branch/output. For a publication-grade 'w/o contrastive' "
        "ablation, silent fallback is not acceptable. Please add a proper switch "
        "to vit_model_gene_wsi_concat.py first."
    )


# -----------------------------------------------------------------------------
# Output normalization helpers
# -----------------------------------------------------------------------------
def _pick_pred_from_out(out: Any, batch_size: int) -> torch.Tensor:
    """
    Robustly extract the survival prediction tensor from model output.

    Preference order:
      1) (B,1) or (B,)
      2) any tensor whose first dimension matches batch size
      3) first tensor found
    """
    if torch.is_tensor(out):
        return out

    if isinstance(out, (tuple, list)):
        for item in out:
            if torch.is_tensor(item) and item.shape[0] == batch_size:
                if (item.ndim == 2 and item.shape[1] == 1) or (item.ndim == 1):
                    return item

        for item in out:
            if torch.is_tensor(item) and item.shape[0] == batch_size:
                return item

        for item in out:
            if torch.is_tensor(item):
                return item

    raise ValueError(f"Cannot extract prediction tensor from output type={type(out)}")


def _to_scalar_tensor(x: Any, device: torch.device) -> torch.Tensor:
    """
    Convert scalar-like value to shape-(1,) tensor on target device.
    Multi-element tensors are reduced by mean.
    """
    if torch.is_tensor(x):
        x = x.to(device)
        if x.numel() > 1:
            x = x.mean()
        return x.reshape(1)
    return torch.tensor(float(x), device=device).reshape(1)


def _try_unpack_apl_losses(out: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Best-effort extraction of APL auxiliary losses from output.

    Supported patterns:
      - (pred, attn, loss_div, loss_bal)
      - (pred, loss_div, loss_bal)

    If not available, returns zeros.
    """
    zero = torch.zeros(1, device=device)

    if not isinstance(out, (tuple, list)):
        return zero, zero

    if len(out) >= 4:
        return _to_scalar_tensor(out[-2], device), _to_scalar_tensor(out[-1], device)

    if len(out) == 3:
        return _to_scalar_tensor(out[-2], device), _to_scalar_tensor(out[-1], device)

    return zero, zero


def _try_unpack_attn(out: Any, pred: torch.Tensor) -> Optional[Any]:
    """
    Optional extraction of attention-like output.

    For the no-contrastive ablation, downstream code does not rely on this
    value, so returning None is acceptable. We only keep it if there is an
    obvious second non-scalar output.
    """
    if not isinstance(out, (tuple, list)):
        return None

    if len(out) < 2:
        return None

    cand = out[1]

    # Avoid mistaking scalar losses for attention.
    if torch.is_tensor(cand):
        if cand.numel() == 1:
            return None
        # If cand is literally the same tensor object as pred, ignore it.
        if cand is pred:
            return None
        return cand

    # Non-tensor structured attention outputs are allowed.
    return cand


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------
class _NoContrastiveAdapter(nn.Module):
    """
    Wrap the base model and normalize forward output to:
        (pred, attn, loss_div, loss_bal)
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x_wsi: torch.Tensor, x_gene: torch.Tensor):
        out = self.base(x_wsi, x_gene)

        batch_size = x_wsi.shape[0] if x_wsi is not None else x_gene.shape[0]
        pred = _pick_pred_from_out(out, batch_size)
        device = pred.device

        attn = _try_unpack_attn(out, pred)
        loss_div, loss_bal = _try_unpack_apl_losses(out, device=device)

        return pred, attn, loss_div, loss_bal


# -----------------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------------
def my_model(
    num_classes: int = 1,
    has_logits: bool = True,
    wsi_block: int = 12,
    gene_block: int = 3,
    dpr: float = 0.1,
    proto_k: int = 0,
    proto_tau: float = 0.07,
):
    """
    Entry point aligned with train_survival.py / call_with_sig().

    Parameters
    ----------
    proto_k <= 0 is interpreted as APL disabled when the base model supports
    an explicit `use_apl` flag.
    """
    use_apl = int(proto_k) > 0

    kwargs = dict(
        num_classes=num_classes,
        has_logits=has_logits,
        wsi_block=wsi_block,
        gene_block=gene_block,
        dpr=dpr,
        proto_k=int(proto_k),
        proto_tau=float(proto_tau),
    )

    # If the base model exposes use_apl, pass it explicitly.
    if _supports_kw(_concat_my_model, "use_apl"):
        kwargs["use_apl"] = use_apl

    # Strictly disable contrastive branch/output.
    contrastive_kw, contrastive_value = _choose_contrastive_off_kw(_concat_my_model)
    kwargs[contrastive_kw] = contrastive_value

    kwargs = _filter_supported_kwargs(_concat_my_model, kwargs)

    base = _concat_my_model(**kwargs)
    return _NoContrastiveAdapter(base)
