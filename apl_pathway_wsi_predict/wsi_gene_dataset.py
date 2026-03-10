#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wsi_gene_dataset.py

Dataset for PRISM survival modeling.

Supported modes via train_flag:
    0: WSI + gene
    1: WSI only
    2: Gene only

Returned sample formats (the last field is always sid=patient_key):
    - train_flag=0: (wsi, gene, futime, fustat, sid)
    - train_flag=1: (wsi, futime, fustat, sid)
    - train_flag=2: (gene, futime, fustat, sid)

Design principles:
1. Keep patient-ID matching robust across TCGA / CPTAC naming variants.
2. Keep valid/test WSI view selection deterministic and reproducible.
3. Do not perform redundant dataset-level shuffling for training;
   training order should be controlled by DataLoader(shuffle=True).
4. Preserve the original public I/O contract used by train_survival.py.
"""

from __future__ import annotations

import os
import glob
import json
import random
import re
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def sample_key(s: str) -> str:
    """
    Normalize various patient/file identifiers into a stable patient key.

    Rules
    -----
    - TCGA: keep first 3 fields -> TCGA-XX-YYYY
      e.g. TCGA-B5-A3S1-01Z-00-DX1 -> TCGA-B5-A3S1
    - C3L/C3N: keep first 2 fields -> C3L-00145 / C3N-00321
      e.g. C3L-00145-21 -> C3L-00145
    - Others: strip suffix, split on '.' / '_', fallback to first 12 chars
    """
    if s is None:
        return ""

    s = os.path.basename(str(s))
    s = re.sub(r"\.(npy|pth|pt|txt|csv|tsv)$", "", s)

    if s.startswith("TCGA-"):
        parts = s.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
        return s[:12]

    if s.startswith("C3L-") or s.startswith("C3N-"):
        parts = s.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
        return s

    s2 = re.split(r"[._]", s)[0]
    return s2[:12] if len(s2) > 12 else s2


class MyDataSet(Dataset):
    """
    PRISM multimodal survival dataset.

    Parameters
    ----------
    wsi_feature_dir : str
        Directory containing WSI feature .txt files.
    gene_feature_dir : str
        Directory containing gene/pathway feature .npy files.
    cox_txt_path : str
        Survival text file with at least 3 columns:
            patient_id  futime  fustat
    max_dim : int
        Kept only for backward compatibility.
    mode : str
        One of {"train", "valid", "test"}.
    transform : callable or None
        Optional transform applied to loaded WSI features.
    train_flag : int
        0 = wsi+gene, 1 = wsi only, 2 = gene only.
    view_seed : int
        Seed used to deterministically choose a fixed WSI view in valid/test mode.
    """

    def __init__(
        self,
        wsi_feature_dir: str,
        gene_feature_dir: str,
        cox_txt_path: str,
        max_dim: int = 10,
        mode: str = "train",
        transform=None,
        train_flag: int = 0,
        view_seed: int = 42,
    ) -> None:
        super().__init__()

        self.wsi_feature_dir = wsi_feature_dir or ""
        self.gene_feature_dir = gene_feature_dir or ""
        self.cox_txt_path = cox_txt_path
        self.max_dim = max_dim
        self.mode = str(mode)
        self.transform = transform
        self.train_flag = int(train_flag)
        self.view_seed = int(view_seed)

        if self.mode not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported mode={self.mode}. Expected one of: train, valid, test")

        if self.train_flag not in {0, 1, 2}:
            raise ValueError(f"Unsupported train_flag={self.train_flag}. Expected one of: 0, 1, 2")

        if not os.path.exists(self.cox_txt_path):
            raise FileNotFoundError(f"cox_txt_path not found: {self.cox_txt_path}")

        if self.train_flag in (0, 1):
            if (not self.wsi_feature_dir) or (not os.path.exists(self.wsi_feature_dir)):
                raise FileNotFoundError(f"WSI feature directory not found: {self.wsi_feature_dir}")

        if self.train_flag in (0, 2):
            if (not self.gene_feature_dir) or (not os.path.exists(self.gene_feature_dir)):
                raise FileNotFoundError(f"Gene feature directory not found: {self.gene_feature_dir}")

        # Feature file lists
        self.gene_file_list = (
            glob.glob(os.path.join(self.gene_feature_dir, "*.npy"))
            if (self.gene_feature_dir and os.path.exists(self.gene_feature_dir))
            else []
        )
        self.wsi_file_list = (
            glob.glob(os.path.join(self.wsi_feature_dir, "*.txt"))
            if (self.wsi_feature_dir and os.path.exists(self.wsi_feature_dir))
            else []
        )

        # gene_map: sid -> filepath
        self.gene_map: Dict[str, str] = {}
        for fp in self.gene_file_list:
            sid = sample_key(fp)
            if sid not in self.gene_map:
                self.gene_map[sid] = fp
            else:
                # Prefer shorter basename if duplicated
                if len(os.path.basename(fp)) < len(os.path.basename(self.gene_map[sid])):
                    self.gene_map[sid] = fp

        # wsi_map: sid -> list[filepaths]
        self.wsi_map: Dict[str, List[str]] = {}
        for fp in self.wsi_file_list:
            sid = sample_key(fp)
            self.wsi_map.setdefault(sid, []).append(fp)

        self.patient_list: List[str] = []
        self.cox_dict: Dict[str, Tuple[float, float]] = {}

        self._pre_process()

    def _pre_process(self) -> None:
        """
        Build:
            - self.cox_dict: sid -> (futime, fustat)
            - self.patient_list: sorted patient keys kept for current train_flag
        """
        cox_ids: List[str] = []

        with open(self.cox_txt_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                raw_id, futime, fustat = parts[0], parts[1], parts[2]
                sid = sample_key(raw_id)
                cox_ids.append(sid)

                # Keep first occurrence only
                if sid not in self.cox_dict:
                    self.cox_dict[sid] = (float(futime), float(fustat))

        cox_set = set(cox_ids)
        wsi_set = set(self.wsi_map.keys())
        gene_set = set(self.gene_map.keys())

        if self.train_flag == 1:          # wsi only
            keep = cox_set.intersection(wsi_set)
        elif self.train_flag == 2:        # gene only
            keep = cox_set.intersection(gene_set)
        else:                             # wsi + gene
            keep = cox_set.intersection(wsi_set, gene_set)

        # Publication-grade behavior:
        # keep dataset order deterministic; let DataLoader(shuffle=True) handle training shuffle.
        self.patient_list = sorted(list(keep))

    def __len__(self) -> int:
        return len(self.patient_list)

    def _select_wsi_file(self, sid: str) -> str:
        """
        Select one WSI feature file for a patient.

        - train: random view selection
        - valid/test: deterministic pseudo-random selection based on sid and view_seed
        """
        if sid not in self.wsi_map:
            raise FileNotFoundError(f"No WSI txt found for sid={sid} in {self.wsi_feature_dir}")

        wsi_files = self.wsi_map[sid]

        if self.mode == "train":
            return random.choice(wsi_files)

        files = sorted(wsi_files)
        h = hashlib.md5(f"{sid}|{self.view_seed}".encode("utf-8")).hexdigest()
        idx = int(h, 16) % len(files)
        return files[idx]

    def __getitem__(self, index: int):
        sid = self.patient_list[index]

        futime, fustat = self.cox_dict[sid]
        futime = float(futime)
        fustat = float(fustat)

        wsi_feat_t: Optional[torch.Tensor] = None
        gene_feat_t: Optional[torch.Tensor] = None

        # -------- WSI branch --------
        if self.train_flag in (0, 1):
            wsi_file = self._select_wsi_file(sid)

            with open(wsi_file, "r") as f_wsi:
                wsi_feat = np.asarray(json.load(f_wsi), dtype=np.float32)

            if self.transform is not None:
                wsi_feat = self.transform(wsi_feat)

            wsi_feat_t = torch.tensor(wsi_feat, dtype=torch.float32)

        # -------- Gene branch --------
        if self.train_flag in (0, 2):
            if sid not in self.gene_map:
                raise FileNotFoundError(f"No gene npy found for sid={sid} in {self.gene_feature_dir}")

            gene_file = self.gene_map[sid]
            gene_feat = np.load(gene_file, allow_pickle=False).astype(np.float32, copy=False)
            gene_feat_t = torch.tensor(gene_feat, dtype=torch.float32)

        # Preserve original output contract
        if self.train_flag == 1:
            return wsi_feat_t, futime, fustat, sid
        elif self.train_flag == 2:
            return gene_feat_t, futime, fustat, sid
        else:
            return wsi_feat_t, gene_feat_t, futime, fustat, sid

    def collate_fn(self, batch):
        """
        Preserve original collate output contract used by utils_cox.py / train_survival.py.
        """
        if self.train_flag == 1:
            wsi_feat, futime, fustat, sid = tuple(zip(*batch))
            wsi_feat = torch.stack(wsi_feat, dim=0)
            futime = torch.tensor(futime, dtype=torch.float32)
            fustat = torch.tensor(fustat, dtype=torch.float32)
            sid = list(sid)
            return wsi_feat, futime, fustat, sid

        elif self.train_flag == 2:
            gene_feat, futime, fustat, sid = tuple(zip(*batch))
            gene_feat = torch.stack(gene_feat, dim=0)
            futime = torch.tensor(futime, dtype=torch.float32)
            fustat = torch.tensor(fustat, dtype=torch.float32)
            sid = list(sid)
            return gene_feat, futime, fustat, sid

        else:
            wsi_feat, gene_feat, futime, fustat, sid = tuple(zip(*batch))
            wsi_feat = torch.stack(wsi_feat, dim=0)
            gene_feat = torch.stack(gene_feat, dim=0)
            futime = torch.tensor(futime, dtype=torch.float32)
            fustat = torch.tensor(fustat, dtype=torch.float32)
            sid = list(sid)
            return wsi_feat, gene_feat, futime, fustat, sid
