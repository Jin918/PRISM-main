#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils_cox_predict.py

Utilities for survival-model inference.

Main responsibilities:
- survival metrics: c-index / median-split accuracy / log-rank p
- robust parsing of model outputs from different PRISM variants
- exporting attention maps and APL assignment matrices
- prediction loop for trained survival models

Supported inference modes:
    train_flag = 0 : wsi + gene
    train_flag = 1 : wsi-only
    train_flag = 2 : gene-only (no attention export)

Expected dataloader batch:
    (wsi_features, gene_features, futime, fustat, file_name)

Notes:
- This module is designed for inference only.
- Loss is computed only for compatibility/reporting and is not required for deployment.
"""

import os
import sys
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test


# =========================================================
# Metrics
# =========================================================
def cindex_legacy(hazards, labels, survtime_all) -> float:
    """
    Legacy pairwise c-index implementation.
    Higher hazard means higher risk.
    """
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(hazards):
        hazards = hazards.detach().cpu().numpy().reshape(-1)
    if torch.is_tensor(survtime_all):
        survtime_all = survtime_all.detach().cpu().numpy().reshape(-1)

    labels = np.asarray(labels).reshape(-1).astype(bool)
    hazards = np.asarray(hazards).reshape(-1)
    survtime_all = np.asarray(survtime_all).reshape(-1)

    concord = 0.0
    total = 0.0
    n = labels.shape[0]

    for i in range(n):
        if labels[i]:
            for j in range(n):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]:
                        concord += 1
                    elif hazards[j] == hazards[i]:
                        concord += 0.5

    return float(concord / total) if total > 0 else 0.0


def cindex_lifeline(hazards, labels, survtime_all) -> float:
    """
    lifelines-based c-index.
    lifelines expects larger score -> longer survival,
    so we pass -hazard as score.
    """
    if torch.is_tensor(hazards):
        hazards_np = hazards.detach().cpu().numpy().reshape(-1)
    else:
        hazards_np = np.asarray(hazards).reshape(-1)

    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy().reshape(-1)
    else:
        labels_np = np.asarray(labels).reshape(-1)

    if torch.is_tensor(survtime_all):
        surv_np = survtime_all.detach().cpu().numpy().reshape(-1)
    else:
        surv_np = np.asarray(survtime_all).reshape(-1)

    keep = ~np.isnan(hazards_np)
    hazards_np = hazards_np[keep]
    labels_np = labels_np[keep]
    surv_np = surv_np[keep]

    if len(hazards_np) == 0:
        return 0.0

    return float(concordance_index(surv_np, -hazards_np, labels_np))


def accuracy_cox(hazards, labels) -> float:
    """
    Median-split survival classification accuracy.
    This is a rough descriptive metric only.
    """
    hazards_np = hazards.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1).astype(int)

    if len(labels_np) == 0:
        return 0.0

    median = np.median(hazards_np)
    pred_group = (hazards_np > median).astype(int)
    return float(np.mean(pred_group == labels_np))


def cox_log_rank(hazards, labels, survtime_all) -> float:
    """
    Log-rank p-value after median dichotomization of predicted hazards.
    """
    hazards_np = hazards.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1).astype(int)
    surv_np = survtime_all.detach().cpu().numpy().reshape(-1)

    median = np.median(hazards_np)
    group = (hazards_np > median).astype(int)

    idx0 = group == 0
    t1, t2 = surv_np[idx0], surv_np[~idx0]
    e1, e2 = labels_np[idx0], labels_np[~idx0]

    if len(t1) == 0 or len(t2) == 0:
        return 1.0

    res = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
    return float(res.p_value)


# =========================================================
# Inference-time Cox loss (optional, for reporting only)
# =========================================================
def cox_loss(survtime, censor, hazard_pred) -> torch.Tensor:
    """
    Cox partial likelihood loss for a batch.
    hazard_pred: (B, 1) or (B,)
    survtime:    (B,)
    censor:      (B,)  1=event, 0=censored
    """
    device = hazard_pred.device

    survtime = survtime.reshape(-1)
    censor = censor.reshape(-1)
    theta = hazard_pred.reshape(-1)

    n = len(survtime)
    surv_np = survtime.detach().cpu().numpy().reshape(-1)

    r_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if surv_np[j] >= surv_np[i]:
                r_mat[i, j] = 1.0

    r_mat = torch.as_tensor(r_mat, dtype=torch.float32, device=device)
    exp_theta = torch.exp(theta)
    denom = torch.sum(exp_theta * r_mat, dim=1).clamp_min(1e-12)

    loss = -torch.mean((theta - torch.log(denom)) * censor)
    return loss


# =========================================================
# File helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def basename_no_ext(x: str) -> str:
    x = str(x)
    base = os.path.basename(x)
    return os.path.splitext(base)[0]


# =========================================================
# Output parsing
# =========================================================
def pick_pred_from_out(out: Any, batch_size: int) -> torch.Tensor:
    """
    Robustly extract hazard prediction tensor from model output.
    Priority:
      1) tensor with shape (B,1) or (B,)
      2) any tensor aligned on batch dimension
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


def parse_multimodal_output(out: Any, batch_size: int):
    """
    Parse outputs from multimodal survival models.

    Returns:
        gene2wsi_feature : Optional[Tensor]
        pred             : Tensor
        attn             : Optional[Tensor]
        assign           : Optional[Tensor]

    Supported patterns include:
        (gene2wsi, pred)
        (gene2wsi, pred, attn)
        (gene2wsi, pred, attn, assign)
        (gene2wsi, pred, loss_div, loss_bal)
        (gene2wsi, pred, loss_div, loss_bal, attn)
        (gene2wsi, pred, loss_div, loss_bal, attn, assign)
    """
    gene2wsi_feature = None
    pred = None
    attn = None
    assign = None

    if torch.is_tensor(out):
        pred = out
        return gene2wsi_feature, pred, attn, assign

    if not isinstance(out, (tuple, list)):
        raise TypeError(f"Model output must be tensor / tuple / list, got {type(out)}")

    if len(out) == 0:
        raise ValueError("Empty model output")

    gene2wsi_feature = out[0] if torch.is_tensor(out[0]) else None
    pred = pick_pred_from_out(out, batch_size)

    for item in out:
        if not torch.is_tensor(item):
            continue

        # attention: usually (B,H,G,K) or (B,G,K)
        if item.shape[0] == batch_size and item.ndim in (3, 4):
            if item is not pred and item is not gene2wsi_feature:
                if attn is None:
                    attn = item
                    continue

        # assign: usually (B,P,K)
        if item.shape[0] == batch_size and item.ndim == 3:
            if item is not pred and item is not gene2wsi_feature and item is not attn:
                assign = item

    return gene2wsi_feature, pred, attn, assign


# =========================================================
# Attention / assignment export
# =========================================================
def save_assign_for_sample(assign_one: torch.Tensor, save_attn_dir: str, base_name: str):
    """
    Save APL assignment matrix once per sample.

    assign_one: (P, K) on CPU
    save path : save_attn_dir/assign/{sample}_assign.pth
    """
    assign_dir = os.path.join(save_attn_dir, "assign")
    ensure_dir(assign_dir)
    out_path = os.path.join(assign_dir, f"{base_name}_assign.pth")
    if not os.path.exists(out_path):
        torch.save(assign_one, out_path)


def save_attention_batch(
    attn: Optional[torch.Tensor],
    assign: Optional[torch.Tensor],
    file_name,
    save_attn_dir: str,
):
    """
    Save attention maps and optional APL assignments.

    attn expected:
        (B, H, G, K_or_P) or (B, G, K_or_P)

    assign expected:
        (B, P, K) or (P, K)
    """
    if attn is None:
        return

    if attn.ndim == 3:
        attn = attn.unsqueeze(1)  # -> (B,1,G,K)

    if attn.ndim != 4:
        return

    attn_cpu = attn.detach().cpu()
    heads = int(attn_cpu.shape[1])

    assign_cpu = None
    if assign is not None and torch.is_tensor(assign):
        if assign.ndim == 2:
            assign_cpu = assign.detach().cpu().unsqueeze(0)
        elif assign.ndim == 3:
            assign_cpu = assign.detach().cpu()

    for head_idx in range(heads):
        head_dir = os.path.join(save_attn_dir, f"head{head_idx}")
        ensure_dir(head_dir)

        single_head = attn_cpu[:, head_idx, :, :]  # (B, G, K_or_P)

        for b in range(single_head.shape[0]):
            base = basename_no_ext(file_name[b])
            attn_path = os.path.join(head_dir, f"{base}.pth")
            torch.save(single_head[b], attn_path)

    if assign_cpu is not None:
        for b in range(assign_cpu.shape[0]):
            base = basename_no_ext(file_name[b])
            save_assign_for_sample(assign_cpu[b], save_attn_dir, base)


# =========================================================
# Main inference loop
# =========================================================
@torch.no_grad()
def predict(
    model,
    topK,
    criterion,
    data_loader,
    json_path,
    save_attn_dir,
    reg_loss=False,
    train_flag=0,
    contrastive_loss_flag=0,
):
    """
    Inference loop.

    Parameters are kept compatible with the original codebase.

    Returns:
        mean_loss, accuracy, logrank_p, c_index
    """
    del topK
    del criterion
    del reg_loss
    del contrastive_loss_flag

    model.eval()
    device = next(model.parameters()).device

    ensure_dir(save_attn_dir)

    accu_loss = torch.zeros(1, device=device)

    pred_all = None
    survtime_all = None
    fustat_all = None

    loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(loader):
        if len(data) != 5:
            raise ValueError(
                "predict() expects dataloader batch = "
                "(wsi_features, gene_features, futime, fustat, file_name)"
            )

        wsi_features, gene_features, futime, fustat, file_name = data

        if torch.is_tensor(wsi_features):
            wsi_features = wsi_features.to(device, non_blocking=True)
        if torch.is_tensor(gene_features):
            gene_features = gene_features.to(device, non_blocking=True)

        futime = futime.to(device, non_blocking=True)
        fustat = fustat.to(device, non_blocking=True)
        batch_size = futime.shape[0]

        if train_flag == 0:
            out = model(wsi_features, gene_features)
            _, pred, attn, assign = parse_multimodal_output(out, batch_size=batch_size)

        elif train_flag == 1:
            out = model(wsi_features)
            pred = pick_pred_from_out(out, batch_size=batch_size)
            attn = None
            assign = None

        elif train_flag == 2:
            out = model(gene_features)
            pred = pick_pred_from_out(out, batch_size=batch_size)
            attn = None
            assign = None

        else:
            raise ValueError(f"predict() supports train_flag in {{0,1,2}}, got {train_flag}")

        loss = cox_loss(futime, fustat, pred).mean()
        accu_loss += loss

        pred_det = pred.detach()
        futime_det = futime.detach()
        fustat_det = fustat.detach()

        if pred_all is None:
            pred_all = pred_det
            survtime_all = futime_det
            fustat_all = fustat_det
        else:
            pred_all = torch.cat([pred_all, pred_det], dim=0)
            survtime_all = torch.cat([survtime_all, futime_det], dim=0)
            fustat_all = torch.cat([fustat_all, fustat_det], dim=0)

        save_attention_batch(
            attn=attn,
            assign=assign,
            file_name=file_name,
            save_attn_dir=save_attn_dir,
        )

        loader.desc = f"[predict] loss: {accu_loss.item() / (step + 1):.6f}"

    if pred_all is None:
        raise RuntimeError("predict(): dataloader yielded 0 batches.")

    acc = accuracy_cox(pred_all, fustat_all)
    pvalue_pred = cox_log_rank(pred_all, fustat_all, survtime_all)
    c_index = cindex_lifeline(pred_all, fustat_all, survtime_all)

    return accu_loss.item() / (step + 1), acc, pvalue_pred, c_index
