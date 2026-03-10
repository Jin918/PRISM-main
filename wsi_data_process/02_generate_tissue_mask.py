#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import logging
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


ID_PATTERNS = [
    re.compile(r"(C3L-\d{5})", re.IGNORECASE),
    re.compile(r"(C3N-\d{5})", re.IGNORECASE),
    re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE),
]


def extract_case_id(s: str):
    s = str(s)
    for pat in ID_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1).upper()

    stem = Path(s).stem
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()
    if len(parts) >= 2 and parts[0].upper().startswith("C3"):
        return "-".join(parts[:2]).upper()
    return None


def _init_logging(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.ERROR,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
    )


def load_ids_from_txt(path: str):
    ids = set()
    if path and os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    ids.add(s.upper())
    return ids


def load_ids_from_csv(path: str, id_col: str):
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"clin_csv not found: {path}")

    df = pd.read_csv(path)
    if id_col not in df.columns:
        for cand in ["ID", "Patient_ID", "Proteomics_Participant_ID"]:
            if cand in df.columns:
                id_col = cand
                break
        else:
            raise ValueError(
                f"Cannot find id_col={id_col} in {path}. "
                f"Available columns (partial): {list(df.columns)[:30]}"
            )

    ids = set()
    for v in df[id_col].astype(str).tolist():
        pid = extract_case_id(v)
        if pid:
            ids.add(pid)
    return ids


def collect_wsi_paths(wsi_root: str, suffix: str):
    paths = []
    for root, _, files in os.walk(wsi_root):
        for file in files:
            if suffix:
                if file.endswith(suffix):
                    paths.append(os.path.join(root, file))
            else:
                if file.endswith(".svs"):
                    paths.append(os.path.join(root, file))
    return sorted(paths)


def suffix_tag(suffix: str):
    if suffix == "-21.svs":
        return "-21"
    if suffix == ".svs":
        return "all"
    if not suffix:
        return "all"
    return suffix.replace(".svs", "").replace("/", "_")


def process_one(path, npy_dir, gray_dir, thumb_dir, level, rgb_min):
    try:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]

        out_npy = os.path.join(npy_dir, stem + ".npy")
        if os.path.exists(out_npy):
            return ("skip", path, "")

        slide = openslide.OpenSlide(path)
        read_level = min(level, slide.level_count - 1)

        img = slide.read_region(
            (0, 0),
            read_level,
            slide.level_dimensions[read_level]
        ).convert("RGB")
        slide.close()

        img.save(os.path.join(thumb_dir, stem + ".png"))

        img_rgb = np.transpose(np.array(img), axes=[1, 0, 2])
        _ = rgb2hsv(img_rgb)  # 保留原始处理口径

        background_r = img_rgb[:, :, 0] > threshold_otsu(img_rgb[:, :, 0])
        background_g = img_rgb[:, :, 1] > threshold_otsu(img_rgb[:, :, 1])
        background_b = img_rgb[:, :, 2] > threshold_otsu(img_rgb[:, :, 2])
        tissue_rgb = np.logical_not(background_r & background_g & background_b)

        min_r = img_rgb[:, :, 0] > rgb_min
        min_g = img_rgb[:, :, 1] > rgb_min
        min_b = img_rgb[:, :, 2] > rgb_min

        tissue_mask = tissue_rgb & min_r & min_g & min_b

        np.save(out_npy, tissue_mask)
        plt.imsave(os.path.join(gray_dir, stem + ".png"), tissue_mask, cmap="gray")

        return ("ok", path, "")

    except Exception as e:
        tb = traceback.format_exc()
        return ("fail", path, f"{e}\n{tb}")


def run(args):
    Path(args.npy_path).mkdir(parents=True, exist_ok=True)
    Path(args.gray_path).mkdir(parents=True, exist_ok=True)
    Path(args.thumb_path).mkdir(parents=True, exist_ok=True)

    clin_ids = set()
    if args.clinical_ids_txt:
        clin_ids |= load_ids_from_txt(args.clinical_ids_txt)
    if args.clin_csv:
        clin_ids |= load_ids_from_csv(args.clin_csv, args.id_col)

    paths = collect_wsi_paths(args.wsi_path, args.suffix)
    print(f"[STATS] Found matched WSI files before filtering: {len(paths)}")

    if clin_ids:
        keep_paths = []
        keep_ids = set()
        seen_ids = set()
        no_id = 0

        for p in paths:
            pid = extract_case_id(os.path.basename(p)) or extract_case_id(p)
            if pid is None:
                no_id += 1
                continue
            seen_ids.add(pid)
            if pid in clin_ids:
                keep_paths.append(p)
                keep_ids.add(pid)

        miss_ids = sorted(list(clin_ids - keep_ids))
        paths = sorted(keep_paths)

        print(f"[STATS] Clinical IDs loaded: {len(clin_ids)}")
        print(f"[STATS] Parsed WSI IDs: {len(seen_ids)} (no_id={no_id})")
        print(f"[STATS] Intersection: {len(keep_ids)} IDs, {len(paths)} files")
        print(f"[STATS] Missing clinical IDs in WSI: {len(miss_ids)}")

        if args.out_base:
            out_base = Path(args.out_base)
            out_base.mkdir(parents=True, exist_ok=True)
            tag = suffix_tag(args.suffix)
            (out_base / f"keep_svs_{tag}.list").write_text(
                "\n".join(paths) + ("\n" if paths else "")
            )
            (out_base / f"keep_ids_{tag}.list").write_text(
                "\n".join(sorted(keep_ids)) + ("\n" if keep_ids else "")
            )
            (out_base / f"missing_{tag}_ids.list").write_text(
                "\n".join(miss_ids) + ("\n" if miss_ids else "")
            )

    if len(paths) == 0:
        print("[WARN] No WSI matched the current settings.")
        return

    log_path = str(Path(args.out_base or args.npy_path) / "error_records.log")
    _init_logging(log_path)

    n_workers = max(1, int(args.workers))
    print(f"Using {n_workers} processes...")

    ok = 0
    skip = 0
    fail = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_logging,
        initargs=(log_path,)
    ) as ex:
        futures = [
            ex.submit(
                process_one,
                p,
                args.npy_path,
                args.gray_path,
                args.thumb_path,
                args.level,
                args.RGB_min
            )
            for p in paths
        ]

        for fu in as_completed(futures):
            status, path, msg = fu.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"[FAIL] {path}")
                logging.error(f"Failed to process file: {path}\n{msg}")

    print(f"[DONE] ok={ok}, skip={skip}, fail={fail}")
    print(f"[OUT] npy={args.npy_path}")
    print(f"[OUT] gray={args.gray_path}")
    print(f"[OUT] thumb={args.thumb_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tissue masks for WSI files"
    )
    parser.add_argument("--wsi_path", required=True, type=str)

    parser.add_argument("--npy_path", required=True, type=str)
    parser.add_argument("--gray_path", required=True, type=str)
    parser.add_argument("--thumb_path", required=True, type=str)

    parser.add_argument(
        "--out_base",
        default="",
        type=str,
        help="optional: save keep/missing lists and log here"
    )

    parser.add_argument("--level", default=2, type=int)
    parser.add_argument("--RGB_min", default=50, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument(
        "--suffix",
        default=".svs",
        type=str,
        help="e.g. '.svs' or '-21.svs'"
    )

    parser.add_argument(
        "--clinical_ids_txt",
        default="",
        type=str,
        help="optional txt file, one case ID per line"
    )
    parser.add_argument(
        "--clin_csv",
        default="",
        type=str,
        help="optional clinical csv used for ID filtering"
    )
    parser.add_argument(
        "--id_col",
        default="ID",
        type=str,
        help="ID column name in clin_csv"
    )

    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    run(args)


if __name__ == "__main__":
    main()
