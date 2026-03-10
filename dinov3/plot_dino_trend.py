#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def infer_epoch_from_name(name: str):
    """
    支持：
      baseline -> 0
      ep4/epoch4/004 -> 4
      checkpoint0006 -> 6
    """
    s = name.lower()
    if "base" in s:
        return 0
    m = re.search(r"(?:ep|epoch|ckpt|checkpoint)?0*([0-9]{1,3})", s)
    if m:
        return int(m.group(1))
    return None


def read_eval_csv(path: str):
    df = pd.read_csv(path)
    # 兼容：有些列不存在（比如 multiclass 没 auc）
    for col in ["val_auc", "val_bal_acc", "val_acc", "train_auc", "train_bal_acc", "train_acc"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def pick_metric(df: pd.DataFrame, task: str, split: str, metric: str):
    """
    split: "val" or "train"
    metric: "auc" or "bal_acc" or "acc"
    """
    col = f"{split}_{metric}"
    sub = df[df["task"] == task]
    if len(sub) == 0:
        return np.nan
    v = sub.iloc[0][col]
    try:
        return float(v)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser("Plot DINOv3 adaptation trend from eval_results_holdout.csv")
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Format: name=csv_path. e.g., baseline=/.../eval_results_holdout.csv ep4=/.../eval_results_holdout.csv",
    )
    ap.add_argument("--out_dir", required=True, help="output dir for plots/csv")
    ap.add_argument("--binary_tasks", default="y12,y24,y36,y60", help="comma-separated binary tasks to plot AUC")
    ap.add_argument(
        "--multi_tasks",
        default="Histologic_Grade,Subtype",
        help="comma-separated multiclass tasks to plot bal-acc",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    binary_tasks = [x.strip() for x in args.binary_tasks.split(",") if x.strip()]
    multi_tasks = [x.strip() for x in args.multi_tasks.split(",") if x.strip()]

    # -------------------------
    # Legend / display name map
    # (only affects plots; DOES NOT affect CSV task lookup)
    # -------------------------
    TASK_LABEL = {
        # binary endpoints
        "y12": "12-month",
        "y24": "24-month",
        "y36": "36-month",
        "y60": "60-month",
        # multiclass / others
        "Histologic_Grade": "Histologic grade",
        "Subtype": "Molecular subtype",
        "mi_50": "MI@50",
    }

    def pretty(t: str) -> str:
        return TASK_LABEL.get(t, t)

    rows = []
    for item in args.runs:
        if "=" not in item:
            raise ValueError(f"Bad runs item: {item}. Use name=csv_path")
        name, path = item.split("=", 1)
        df = read_eval_csv(path)
        ep = infer_epoch_from_name(name)
        if ep is None:
            raise ValueError(
                f"Cannot infer epoch from name='{name}'. Please name it like baseline/ep4/ep6/ep8."
            )

        row = {"name": name, "epoch": ep, "csv": path}
        # binary: val_auc
        for t in binary_tasks:
            row[f"{t}_val_auc"] = pick_metric(df, t, "val", "auc")
            row[f"{t}_train_auc"] = pick_metric(df, t, "train", "auc")
            row[f"{t}_val_balacc"] = pick_metric(df, t, "val", "bal_acc")
        # multiclass: val_bal_acc
        for t in multi_tasks:
            row[f"{t}_val_balacc"] = pick_metric(df, t, "val", "bal_acc")
            row[f"{t}_train_balacc"] = pick_metric(df, t, "train", "bal_acc")
        rows.append(row)

    summ = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    out_csv = os.path.join(args.out_dir, "trend_metrics.csv")
    summ.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # ---- Plot 1: binary AUC trend (val) ----
    plt.figure(figsize=(10, 5))
    for t in binary_tasks:
        plt.plot(
            summ["epoch"],
            summ[f"{t}_val_auc"],
            marker="o",
            label=f"{pretty(t)} (val AUC)",
        )
    plt.xlabel("Adaptation epoch (baseline=0)")
    plt.ylabel("AUC (validation)")
    plt.title("DINOv3 adaptation trend (patient-level linear probe)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png = os.path.join(args.out_dir, "trend_auc.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"[saved] {out_png}")
    plt.close()

    # ---- Plot 2: multiclass bal-acc trend (val) ----
    plt.figure(figsize=(10, 5))
    for t in multi_tasks:
        plt.plot(
            summ["epoch"],
            summ[f"{t}_val_balacc"],
            marker="o",
            label=f"{pretty(t)} (val bal-acc)",
        )
    plt.xlabel("Adaptation epoch (baseline=0)")
    plt.ylabel("Balanced Accuracy (validation)")
    plt.title("DINOv3 adaptation trend (multiclass proxy)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png2 = os.path.join(args.out_dir, "trend_balacc.png")
    plt.tight_layout()
    plt.savefig(out_png2, dpi=200)
    print(f"[saved] {out_png2}")
    plt.close()

    # ---- Optional: print quick suggestion ----
    # example: choose epoch maximizing mean(y36,y60) val_auc
    if "y36_val_auc" in summ.columns and "y60_val_auc" in summ.columns:
        summ["mean_y36y60"] = summ[["y36_val_auc", "y60_val_auc"]].mean(axis=1)
        best = summ.iloc[int(np.nanargmax(summ["mean_y36y60"].values))]
        print(
            f"[auto] best by mean(y36,y60) val AUC: epoch={best['epoch']} name={best['name']} mean={best['mean_y36y60']:.4f}"
        )


if __name__ == "__main__":
    main()
