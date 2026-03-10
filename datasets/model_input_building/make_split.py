#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np


def read_cox(path):
    ids, times, events = [], [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            sid = parts[0]
            t = float(parts[1])
            e = int(float(parts[2]))
            ids.append(sid)
            times.append(t)
            events.append(e)
    ids = np.asarray(ids)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    return ids, times, events


def stratified_split(ids, times, events, val_frac, seed):
    rng = np.random.RandomState(seed)

    idx0 = np.where(events == 0)[0]
    idx1 = np.where(events == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_val = int(np.round(len(idx0) * val_frac))
    n1_val = int(np.round(len(idx1) * val_frac))

    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    tr_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])

    rng.shuffle(val_idx)
    rng.shuffle(tr_idx)
    return tr_idx, val_idx


def write_cox(path, ids, times, events, idx):
    with open(path, "w") as f:
        for i in idx:
            f.write(f"{ids[i]}\t{times[i]:.6f}\t{int(events[i])}\n")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    datasets_dir = repo_root / "datasets"

    default_cox_all = datasets_dir / "model_inputs" / "pamt_287" / "cox" / "all.txt"
    default_out_dir = datasets_dir / "model_inputs" / "pamt_287" / "split_seed302"

    ap = argparse.ArgumentParser()
    ap.add_argument("--cox_all", default=str(default_cox_all), help="TCGA all.txt (id time event)")
    ap.add_argument("--out_dir", default=str(default_out_dir), help="output dir for train.txt/val.txt")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=302, help="split seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, times, events = read_cox(args.cox_all)
    tr_idx, va_idx = stratified_split(ids, times, events, args.val_frac, args.seed)

    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"

    write_cox(train_path, ids, times, events, tr_idx)
    write_cox(val_path, ids, times, events, va_idx)

    def stat(idx):
        n = len(idx)
        ev = int(events[idx].sum())
        return n, ev, ev / max(1, n)

    ntr, evtr, rtr = stat(tr_idx)
    nva, evva, rva = stat(va_idx)

    print(f"[OK] seed={args.seed} val_frac={args.val_frac}")
    print(f" train: n={ntr} events={evtr} rate={rtr:.4f} -> {train_path}")
    print(f" valid: n={nva} events={evva} rate={rva:.4f} -> {val_path}")


if __name__ == "__main__":
    main()
