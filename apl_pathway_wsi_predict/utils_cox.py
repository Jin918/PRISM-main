#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils_cox.py

Core utilities for PRISM survival training and evaluation.

Main features
-------------
- Cox loss and survival-related evaluation metrics
- train_one_epoch / evaluate supporting:
    0: WSI + gene
    1: WSI-only
    2: gene-only
- Optional contrastive loss for multimodal training
- Optional APL regularization terms (loss_div / loss_bal)
- Optional prediction TSV export for downstream R plotting
- Optional IPCW-based time-dependent ROC/AUC evaluation

Notes
-----
- `accuracy_cox()` is only an auxiliary median-split monitoring metric and is
  not a primary survival analysis metric.
- This file is intentionally kept as a drop-in replacement for the existing
  training code, so the main training/evaluation logic is preserved.
"""

import os
import sys
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Basic survival metrics
# -----------------------------------------------------------------------------
def CIndex(hazards: torch.Tensor, labels: torch.Tensor, survtime_all: torch.Tensor) -> float:
    """
    Manual concordance index implementation kept for compatibility.
    Not used as the primary c-index function in the current workflow.
    """
    labels_np = labels.detach().cpu().numpy().reshape(-1).astype(bool)
    hazards_np = hazards.detach().cpu().numpy().reshape(-1)
    surv_np = survtime_all.detach().cpu().numpy().reshape(-1)

    concord = 0.0
    total = 0.0
    n_test = labels_np.shape[0]

    for i in range(n_test):
        if labels_np[i]:
            for j in range(n_test):
                if surv_np[j] > surv_np[i]:
                    total += 1.0
                    if hazards_np[j] < hazards_np[i]:
                        concord += 1.0
                    elif hazards_np[j] == hazards_np[i]:
                        concord += 0.5

    return float(concord / total) if total > 0 else 0.0


def CIndex_lifeline(hazards: torch.Tensor, labels: torch.Tensor, survtime_all: torch.Tensor) -> float:
    """
    Harrell's c-index via lifelines.

    Parameters
    ----------
    hazards : predicted risk scores; higher means higher risk / shorter survival
    labels  : event indicator (1=event, 0=censor)
    survtime_all : follow-up time
    """
    labels_np = labels.detach().cpu().numpy().reshape(-1)
    hazards_np = hazards.detach().cpu().numpy().reshape(-1)
    surv_np = survtime_all.detach().cpu().numpy().reshape(-1)

    keep = ~np.isnan(hazards_np)
    if keep.sum() == 0:
        return 0.0

    # lifelines concordance_index assumes larger score => longer survival,
    # so we use -hazard as the ranking score.
    return float(concordance_index(surv_np[keep], -hazards_np[keep], labels_np[keep]))


def accuracy_cox(hazards: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Auxiliary median-split accuracy for coarse monitoring only.
    This is NOT a primary survival evaluation metric.
    """
    hazards_np = hazards.detach().cpu().numpy().reshape(-1)
    median = np.median(hazards_np)

    pred_group = np.zeros(len(hazards_np), dtype=int)
    pred_group[hazards_np > median] = 1

    labels_np = labels.detach().cpu().numpy().reshape(-1)
    correct = np.sum(pred_group == labels_np)
    return float(correct / max(len(labels_np), 1))


def cox_log_rank(hazards: torch.Tensor, labels: torch.Tensor, survtime_all: torch.Tensor) -> float:
    """
    Log-rank p value after median dichotomization of predicted risk.
    Used as an auxiliary descriptive statistic.
    """
    hazards_np = hazards.detach().cpu().numpy().reshape(-1)
    median = np.median(hazards_np)

    pred_group = np.zeros(len(hazards_np), dtype=int)
    pred_group[hazards_np > median] = 1

    surv_np = survtime_all.detach().cpu().numpy().reshape(-1)
    labels_np = labels.detach().cpu().numpy().reshape(-1)

    low_mask = pred_group == 0
    t1, t2 = surv_np[low_mask], surv_np[~low_mask]
    e1, e2 = labels_np[low_mask], labels_np[~low_mask]

    if len(t1) == 0 or len(t2) == 0:
        return 1.0

    results = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
    return float(results.p_value)


# -----------------------------------------------------------------------------
# Cox loss
# -----------------------------------------------------------------------------
def CoxLoss(
    survtime: torch.Tensor,
    event: torch.Tensor,
    hazard_pred: torch.Tensor,
    loss: str = "cox-nnet",
    model: Optional[nn.Module] = None,
    l2_reg: float = 1e-2,
) -> torch.Tensor:
    """
    Negative partial log-likelihood for Cox PH.

    Parameters
    ----------
    survtime : (B,) or (B,1)
    event    : (B,) or (B,1), 1=event, 0=censor
    hazard_pred : (B,) or (B,1), predicted log-risk / risk score

    Notes
    -----
    - The current implementation intentionally preserves the original risk-set
      construction logic for reproducibility with existing experiments.
    - `loss`, `model`, `l2_reg` are kept in the signature for compatibility,
      although the current workflow uses the standard Cox loss branch only.
    """
    if loss != "cox-nnet":
        raise NotImplementedError(
            "Only 'cox-nnet' branch is retained in this publication-grade version."
        )

    device = hazard_pred.device
    survtime = survtime.reshape(-1)
    event = event.reshape(-1)
    theta = hazard_pred.reshape(-1)

    batch_len = survtime.shape[0]
    surv_np = survtime.detach().cpu().numpy().reshape(-1)

    # Keep the original explicit risk-set construction for reproducibility.
    risk_mat = np.zeros((batch_len, batch_len), dtype=np.float32)
    for i in range(batch_len):
        for j in range(batch_len):
            if surv_np[j] >= surv_np[i]:
                risk_mat[i, j] = 1.0

    risk_mat = torch.from_numpy(risk_mat).to(device=device)

    exp_theta = torch.exp(theta)
    denom = torch.sum(exp_theta * risk_mat, dim=1).clamp_min(1e-12)
    loss_cox = -torch.mean((theta - torch.log(denom)) * event)
    return loss_cox


# -----------------------------------------------------------------------------
# Batch unpack helpers
# -----------------------------------------------------------------------------
def _unpack_batch(data: Any, train_flag: int):
    """
    Standardize dataloader outputs to:
        (wsi_features, gene_features, futime, fustat, sid)

    Supported dataset formats
    -------------------------
    Newer datasets with sid:
        flag=0: (wsi, gene, futime, fustat, sid)
        flag=1: (wsi, futime, fustat, sid)
        flag=2: (gene, futime, fustat, sid)

    Older datasets without sid:
        flag=0: (wsi, gene, futime, fustat)
        flag=1: (wsi, futime, fustat)
        flag=2: (gene, futime, fustat)
    """
    sid = None

    if train_flag == 0:
        if len(data) == 5:
            wsi, gene, futime, fustat, sid = data
        else:
            wsi, gene, futime, fustat = data
        return wsi, gene, futime, fustat, sid

    if train_flag == 1:
        if len(data) == 4:
            wsi, futime, fustat, sid = data
        else:
            wsi, futime, fustat = data
        return wsi, None, futime, fustat, sid

    if train_flag == 2:
        if len(data) == 4:
            gene, futime, fustat, sid = data
        else:
            gene, futime, fustat = data
        return None, gene, futime, fustat, sid

    raise ValueError(f"Unknown train_flag={train_flag}")


# -----------------------------------------------------------------------------
# Forward helpers
# -----------------------------------------------------------------------------
def _get_device_from_model(model: nn.Module) -> torch.device:
    base = model.module if hasattr(model, "module") else model
    return next(base.parameters()).device


def _as_tensor_on(x: Any, device: torch.device) -> torch.Tensor:
    """
    Convert a scalar-like object to a scalar tensor on target device.
    If x is a multi-element tensor, its mean is used.
    """
    if torch.is_tensor(x):
        x = x.to(device)
        if x.numel() > 1:
            x = x.mean()
        return x
    return torch.tensor(float(x), device=device)


def _pick_pred_from_out(out: Any, batch_size: int) -> torch.Tensor:
    """
    Robustly select the prediction tensor from model output.

    Preference order
    ----------------
    1. Tensor with shape (B,1) or (B,)
    2. Any tensor whose first dimension matches batch size
    3. First tensor found in tuple/list
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

    raise ValueError(f"Cannot pick prediction tensor from model output type={type(out)}")


def _unpack_contrastive_out(out: Any) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
    """
    Expected multimodal contrastive output patterns:
      - (gene2wsi_feature, pred)
      - (gene2wsi_feature, pred, loss_div, loss_bal)

    Returns
    -------
    gene2wsi_feature, pred, loss_div, loss_bal
    """
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError(
            f"Contrastive branch expects tuple/list with >=2 elements, got {type(out)}"
        )

    gene2wsi_feature = out[0]
    pred = out[1]
    loss_div = out[2] if len(out) > 2 else 0.0
    loss_bal = out[3] if len(out) > 3 else 0.0

    if torch.is_tensor(loss_div) and loss_div.numel() > 1:
        loss_div = loss_div.mean()
    if torch.is_tensor(loss_bal) and loss_bal.numel() > 1:
        loss_bal = loss_bal.mean()

    return gene2wsi_feature, pred, loss_div, loss_bal


def _try_unpack_apl_losses(out: Any) -> Tuple[Any, Any]:
    """
    Compatibility helper:
    if model output is a tuple/list with length >= 3, interpret the last two
    entries as (loss_div, loss_bal).

    Supported patterns include, for example:
      - (gene2wsi_feature, pred, loss_div, loss_bal, ...)
      - (pred, attn, loss_div, loss_bal)
      - (pred, loss_div, loss_bal)
    """
    loss_div, loss_bal = 0.0, 0.0
    if isinstance(out, (tuple, list)) and len(out) >= 3:
        loss_div, loss_bal = out[-2], out[-1]

        if torch.is_tensor(loss_div) and loss_div.numel() > 1:
            loss_div = loss_div.mean()
        if torch.is_tensor(loss_bal) and loss_bal.numel() > 1:
            loss_bal = loss_bal.mean()

    return loss_div, loss_bal


def _write_pred_tsv(
    save_path: str,
    sid_all: List[str],
    time_all: np.ndarray,
    event_all: np.ndarray,
    risk_all: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("sid\ttime\tevent\trisk\n")
        for sid, t, e, r in zip(sid_all, time_all, event_all, risk_all):
            f.write(f"{sid}\t{float(t)}\t{int(e)}\t{float(r)}\n")


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    topK: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    epoch: int,
    reg_loss=False,
    train_flag: int = 0,
    contrastive_loss_flag: int = 0,
    lambda_div: float = 0.0,
    lambda_bal: float = 0.0,
):
    model.train()
    device = _get_device_from_model(model)

    accu_loss = torch.zeros(1, device=device)
    pred_all = None

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        wsi_features, gene_features, futime, fustat, sid = _unpack_batch(data, train_flag)

        if wsi_features is not None and torch.is_tensor(wsi_features):
            wsi_features = wsi_features.to(device, non_blocking=True)
        if gene_features is not None and torch.is_tensor(gene_features):
            gene_features = gene_features.to(device, non_blocking=True)

        futime = futime.to(device, non_blocking=True) if torch.is_tensor(futime) else futime
        fustat = fustat.to(device, non_blocking=True) if torch.is_tensor(fustat) else fustat

        batch_size = futime.shape[0]

        gene2wsiloss = 0.0
        loss_div, loss_bal = 0.0, 0.0

        # ------------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------------
        if train_flag == 0:
            if contrastive_loss_flag == 1:
                out = model(wsi_features, gene_features)
                gene2wsi_feature, pred, loss_div, loss_bal = _unpack_contrastive_out(out)

                # gene2wsi_feature: (B, N_gene, M_wsi) -> (B, M_wsi, N_gene)
                gene2wsi_feature = gene2wsi_feature.transpose(-2, -1)
                bsz, m_wsi, n_gene = gene2wsi_feature.shape

                k_eff = min(topK, m_wsi)
                labels = torch.zeros(bsz, m_wsi, n_gene, device=device)
                labels[:, :k_eff, :] = 1.0

                sorted_feat, _ = torch.sort(gene2wsi_feature, descending=True, dim=1)
                gene2wsiloss = criterion(sorted_feat, labels)
            else:
                out = model(wsi_features, gene_features)
                pred = _pick_pred_from_out(out, batch_size)
                loss_div, loss_bal = _try_unpack_apl_losses(out)

        elif train_flag == 1:
            out = model(wsi_features)
            pred = _pick_pred_from_out(out, batch_size)
            loss_div, loss_bal = _try_unpack_apl_losses(out)

        elif train_flag == 2:
            out = model(gene_features)
            pred = _pick_pred_from_out(out, batch_size)
            loss_div, loss_bal = 0.0, 0.0

        else:
            raise ValueError(f"Unknown train_flag={train_flag}")

        # ------------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------------
        use_reg = callable(reg_loss)
        base = model.module if hasattr(model, "module") else model

        if train_flag == 0:
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss
        else:
            loss = CoxLoss(futime, fustat, pred)

        if train_flag in (0, 1) and (lambda_div != 0.0 or lambda_bal != 0.0):
            loss = (
                loss
                + float(lambda_div) * _as_tensor_on(loss_div, device)
                + float(lambda_bal) * _as_tensor_on(loss_bal, device)
            )

        if use_reg:
            loss = loss + reg_loss(base)

        loss = loss.mean()

        if epoch == 0 and step == 0:
            cox_loss_val = CoxLoss(futime, fustat, pred).detach().item()
            div_val = loss_div.detach().item() if torch.is_tensor(loss_div) else float(loss_div)
            bal_val = loss_bal.detach().item() if torch.is_tensor(loss_bal) else float(loss_bal)
            print(
                f"[DEBUG] cox={cox_loss_val:.6f} div={div_val:.10f} bal={bal_val:.10f} "
                f"lambda_div={lambda_div} lambda_bal={lambda_bal}"
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accu_loss += loss.detach()
        data_loader.desc = f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.6f}"

        if not torch.isfinite(loss):
            print("WARNING: non-finite loss, ending training", loss)
            sys.exit(1)

        pred_det = pred.detach()
        futime_det = futime.detach()
        fustat_det = fustat.detach()

        if step == 0:
            pred_all = pred_det
            survtime_torch = futime_det
            fustat_torch = fustat_det
        else:
            pred_all = torch.cat([pred_all, pred_det])
            survtime_torch = torch.cat([survtime_torch, futime_det])
            fustat_torch = torch.cat([fustat_torch, fustat_det])

    acc = accuracy_cox(pred_all.data, fustat_torch)
    pvalue_pred = cox_log_rank(pred_all.data, fustat_torch, survtime_torch)
    c_index = CIndex_lifeline(pred_all.data, fustat_torch, survtime_torch)

    return accu_loss.item() / (step + 1), acc, pvalue_pred, c_index


@torch.no_grad()
def evaluate(
    model: nn.Module,
    topK: int,
    criterion: nn.Module,
    data_loader,
    epoch: int,
    json_path: str,
    reg_loss=False,
    train_flag: int = 0,
    contrastive_loss_flag: int = 0,
    lambda_div: float = 0.0,
    lambda_bal: float = 0.0,
    split_name: str = "valid",
    train_times_ipcw=None,
    train_events_ipcw=None,
    roc_times=None,
    save_roc_path=None,
    save_pred_path: Optional[str] = None,
):
    model.eval()
    device = _get_device_from_model(model)

    accu_loss = torch.zeros(1, device=device)
    pred_all = None
    sid_all: List[str] = []

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        wsi_features, gene_features, futime, fustat, sid = _unpack_batch(data, train_flag)

        if sid is not None:
            sid_all.extend([str(x) for x in list(sid)])

        if wsi_features is not None and torch.is_tensor(wsi_features):
            wsi_features = wsi_features.to(device, non_blocking=True)
        if gene_features is not None and torch.is_tensor(gene_features):
            gene_features = gene_features.to(device, non_blocking=True)

        futime = futime.to(device, non_blocking=True) if torch.is_tensor(futime) else futime
        fustat = fustat.to(device, non_blocking=True) if torch.is_tensor(fustat) else fustat

        batch_size = futime.shape[0]

        gene2wsiloss = 0.0
        loss_div, loss_bal = 0.0, 0.0

        # ------------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------------
        if train_flag == 0:
            if contrastive_loss_flag == 1:
                out = model(wsi_features, gene_features)
                gene2wsi_feature, pred, loss_div, loss_bal = _unpack_contrastive_out(out)

                gene2wsi_feature = gene2wsi_feature.transpose(-2, -1)
                bsz, m_wsi, n_gene = gene2wsi_feature.shape

                k_eff = min(topK, m_wsi)
                labels = torch.zeros(bsz, m_wsi, n_gene, device=device)
                labels[:, :k_eff, :] = 1.0

                sorted_feat, _ = torch.sort(gene2wsi_feature, descending=True, dim=1)
                gene2wsiloss = criterion(sorted_feat, labels)
            else:
                out = model(wsi_features, gene_features)
                pred = _pick_pred_from_out(out, batch_size)
                loss_div, loss_bal = _try_unpack_apl_losses(out)

        elif train_flag == 1:
            out = model(wsi_features)
            pred = _pick_pred_from_out(out, batch_size)
            loss_div, loss_bal = _try_unpack_apl_losses(out)

        elif train_flag == 2:
            out = model(gene_features)
            pred = _pick_pred_from_out(out, batch_size)
            loss_div, loss_bal = 0.0, 0.0

        else:
            raise ValueError(f"Unknown train_flag={train_flag}")

        # ------------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------------
        use_reg = callable(reg_loss)
        base = model.module if hasattr(model, "module") else model

        if train_flag == 0:
            loss = CoxLoss(futime, fustat, pred) + gene2wsiloss
        else:
            loss = CoxLoss(futime, fustat, pred)

        if train_flag in (0, 1) and (lambda_div != 0.0 or lambda_bal != 0.0):
            loss = (
                loss
                + float(lambda_div) * _as_tensor_on(loss_div, device)
                + float(lambda_bal) * _as_tensor_on(loss_bal, device)
            )

        if use_reg:
            loss = loss + reg_loss(base)

        loss = loss.mean()
        accu_loss += loss.detach()

        data_loader.desc = f"[{split_name} epoch {epoch}] loss: {accu_loss.item() / (step + 1):.6f}"

        pred_det = pred.detach()
        futime_det = futime.detach()
        fustat_det = fustat.detach()

        if step == 0:
            pred_all = pred_det
            survtime_torch = futime_det
            fustat_torch = fustat_det
        else:
            pred_all = torch.cat([pred_all, pred_det])
            survtime_torch = torch.cat([survtime_torch, futime_det])
            fustat_torch = torch.cat([fustat_torch, fustat_det])

    acc = accuracy_cox(pred_all.data, fustat_torch)
    pvalue_pred = cox_log_rank(pred_all.data, fustat_torch, survtime_torch)
    c_index = CIndex_lifeline(pred_all.data, fustat_torch, survtime_torch)

    # ----------------------------------------------------------------------
    # Optional prediction export for downstream R scripts
    # ----------------------------------------------------------------------
    if save_pred_path is not None:
        risk_scores = pred_all.detach().cpu().numpy().reshape(-1)
        times = survtime_torch.detach().cpu().numpy().reshape(-1).astype(float)
        events = fustat_torch.detach().cpu().numpy().reshape(-1).astype(int)

        if len(sid_all) != len(risk_scores):
            sid_all = [f"row{i}" for i in range(len(risk_scores))]

        _write_pred_tsv(save_pred_path, sid_all, times, events, risk_scores)

    # ----------------------------------------------------------------------
    # Optional IPCW time-dependent ROC
    # ----------------------------------------------------------------------
    tdroc = None
    if (roc_times is not None) and (train_times_ipcw is not None) and (train_events_ipcw is not None):
        risk_scores = pred_all.detach().cpu().numpy().reshape(-1)
        test_times = survtime_torch.detach().cpu().numpy().reshape(-1).astype(float)
        test_events = fustat_torch.detach().cpu().numpy().reshape(-1).astype(int)

        tdroc = time_dependent_roc_ipcw(
            train_times=train_times_ipcw,
            train_events=train_events_ipcw,
            test_times=test_times,
            test_events=test_events,
            risk_scores=risk_scores,
            eval_times=roc_times,
        )

        if save_roc_path:
            plot_tdroc(tdroc, save_roc_path, title=f"{split_name} time-dependent ROC (epoch {epoch})")

    return accu_loss.item() / (step + 1), acc, pvalue_pred, c_index, tdroc


# -----------------------------------------------------------------------------
# IPCW time-dependent ROC helpers
# -----------------------------------------------------------------------------
def _km_surv(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kaplan-Meier survival curve as a right-continuous step function.

    Parameters
    ----------
    times : observed times
    events : 1=event, 0=censor
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    order = np.argsort(times)
    times = times[order]
    events = events[order]

    uniq = np.unique(times)
    n = len(times)

    step_t = [0.0]
    step_s = [1.0]
    s = 1.0

    for t in uniq:
        left = np.searchsorted(times, t, side="left")
        right = np.searchsorted(times, t, side="right")
        n_risk = n - left
        d = events[left:right].sum()

        if n_risk > 0:
            s *= (1.0 - d / n_risk)

        step_t.append(float(t))
        step_s.append(float(s))

    return np.asarray(step_t), np.asarray(step_s)


def _step_eval(step_t: np.ndarray, step_s: np.ndarray, t: float) -> float:
    """Evaluate a right-continuous step function S(t)."""
    idx = np.searchsorted(step_t, t, side="right") - 1
    idx = max(idx, 0)
    return float(step_s[idx])


def time_dependent_roc_ipcw(
    train_times,
    train_events,
    test_times,
    test_events,
    risk_scores,
    eval_times,
):
    """
    Cumulative/dynamic time-dependent ROC with IPCW.

    Procedure
    ---------
    - Estimate censoring distribution G(t)=P(C>t) from the training set
    - Construct case/control labels on the test set at each evaluation time
    - Compute IPCW-weighted ROC and AUC

    Returns
    -------
    dict[t] = {"fpr": ..., "tpr": ..., "auc": ...}
    """
    train_times = np.asarray(train_times, dtype=float)
    train_events = np.asarray(train_events, dtype=int)
    test_times = np.asarray(test_times, dtype=float)
    test_events = np.asarray(test_events, dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)

    censor_events = 1 - train_events
    g_t, g_s = _km_surv(train_times, censor_events)

    def G(x: float) -> float:
        eps = 1e-6
        return max(_step_eval(g_t, g_s, x), eps)

    out = {}
    for t in eval_times:
        t = float(t)

        is_case = (test_events == 1) & (test_times <= t)
        is_ctrl = test_times > t

        mask = is_case | is_ctrl
        if mask.sum() < 5 or is_case.sum() < 2 or is_ctrl.sum() < 2:
            out[t] = {
                "fpr": np.array([0.0, 1.0]),
                "tpr": np.array([0.0, 1.0]),
                "auc": float("nan"),
            }
            continue

        y = np.zeros(mask.sum(), dtype=int)
        y[is_case[mask]] = 1
        s = risk_scores[mask]

        w = np.zeros(mask.sum(), dtype=float)
        case_times = test_times[mask][y == 1]
        w[y == 1] = np.array([1.0 / G(x) for x in case_times], dtype=float)
        w[y == 0] = 1.0 / G(t)

        fpr, tpr, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
        auc = roc_auc_score(y, s, sample_weight=w)

        out[t] = {"fpr": fpr, "tpr": tpr, "auc": float(auc)}

    return out


def plot_tdroc(roc_dict, save_path: str, title: str = "Time-dependent ROC") -> None:
    plt.figure()
    for t, d in roc_dict.items():
        auc = d["auc"]
        plt.plot(d["fpr"], d["tpr"], label=f"t={t:g}, AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
