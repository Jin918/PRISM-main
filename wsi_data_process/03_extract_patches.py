#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import logging
import argparse
import re
from pathlib import Path
from multiprocessing import Pool, Value, Lock

import numpy as np
import pandas as pd
import openslide


SLIDE = None
PATCH_DIR = None
PATCH_SIZE_REAL = None
CUR_LEVEL = None
WSI_NAME = None

count = Value("i", 0)
lock = Lock()


def extract_case_id(s: str):
    s = str(s)

    m = re.search(r"(C3L-\d{5})", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"(C3N-\d{5})", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    stem = Path(s).stem
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()
    if len(parts) >= 2 and parts[0].upper().startswith("C3"):
        return "-".join(parts[:2]).upper()
    return None


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


def fast_count_png(folder: str) -> int:
    try:
        return sum(
            1 for e in os.scandir(folder)
            if e.is_file() and e.name.endswith(".png")
        )
    except FileNotFoundError:
        return 0


def init_worker(wsi_path, patch_dir, patch_size_real, cur_level):
    global SLIDE, PATCH_DIR, PATCH_SIZE_REAL, CUR_LEVEL, WSI_NAME
    SLIDE = openslide.OpenSlide(wsi_path)
    PATCH_DIR = patch_dir
    PATCH_SIZE_REAL = patch_size_real
    CUR_LEVEL = cur_level
    WSI_NAME = Path(wsi_path).stem


def process(coord):
    x, y = coord
    try:
        save_path = os.path.join(PATCH_DIR, f"{WSI_NAME}_{x}_{y}.png")

        if os.path.exists(save_path):
            return

        img = SLIDE.read_region(
            (x, y),
            CUR_LEVEL,
            (PATCH_SIZE_REAL, PATCH_SIZE_REAL)
        ).convert("RGB")

        if PATCH_SIZE_REAL != PATCH_SIZE:
            img = img.resize((PATCH_SIZE, PATCH_SIZE))

        img.save(save_path)

        global count, lock
        with lock:
            count.value += 1
            if count.value % 100 == 0:
                logging.info(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"{count.value} NEW patches generated"
                )

    except Exception as e:
        logging.warning(f"[SKIP] {WSI_NAME} x={x}, y={y}, error={e}")


def run(args):
    global PATCH_SIZE
    PATCH_SIZE = args.patch_size

    logging.basicConfig(level=logging.INFO)

    Path(args.patch_root).mkdir(parents=True, exist_ok=True)

    filter_ids = set()
    if args.clinical_ids_txt:
        filter_ids |= load_ids_from_txt(args.clinical_ids_txt)
    if args.clin_csv:
        filter_ids |= load_ids_from_csv(args.clin_csv, args.id_col)

    svs_files = collect_wsi_paths(args.wsi_root, args.suffix)

    if filter_ids:
        kept = []
        for p in svs_files:
            pid = extract_case_id(os.path.basename(p)) or extract_case_id(p)
            if pid is not None and pid in filter_ids:
                kept.append(p)
        svs_files = kept

    print(f"Found {len(svs_files)} WSI files")

    for idx, wsi_path in enumerate(svs_files):
        wsi_name = Path(wsi_path).stem
        print(f"\n[{idx + 1}/{len(svs_files)}] Processing {wsi_name}")

        patch_dir = os.path.join(args.patch_root, wsi_name)
        os.makedirs(patch_dir, exist_ok=True)

        done_flag = os.path.join(patch_dir, ".done")
        if os.path.exists(done_flag):
            print("Already done (.done exists), skip")
            continue

        mask_file = os.path.join(args.mask_root, wsi_name + ".npy")
        if not os.path.exists(mask_file):
            print("Mask not found, skip")
            continue

        mask = np.load(mask_file)
        slide = openslide.OpenSlide(wsi_path)

        try:
            mpp = float(
                slide.properties.get(
                    openslide.PROPERTY_NAME_MPP_X, 0.0
                ) or 0.0
            )
        except Exception:
            mpp = 0.0

        if 0.08 <= mpp < 0.15:
            max_mag = 80
        elif 0.15 <= mpp < 0.40:
            max_mag = 40
        elif 0.40 <= mpp < 0.70:
            max_mag = 20
        else:
            print(f"Unsupported MPP {mpp}, skip")
            slide.close()
            continue

        rate = max_mag // args.target_level

        if slide.level_count > 1:
            denom = slide.level_downsamples[1]
        else:
            denom = 1.0

        cur_level = round(rate / denom)
        cur_level = min(cur_level, slide.level_count - 1)

        patch_level = round(slide.level_dimensions[0][0] / mask.shape[0])
        patch_size_real = round(rate / slide.level_downsamples[cur_level]) * args.patch_size
        step = int(args.patch_size / patch_level)

        slide.close()

        xs, ys = np.where(mask)
        coords = list(set(zip(
            (xs / rate / step).astype(int),
            (ys / rate / step).astype(int)
        )))
        random.shuffle(coords)

        coords = [
            (
                int((x + 0.5) * patch_level * step * rate - args.patch_size * rate / 2),
                int((y + 0.5) * patch_level * step * rate - args.patch_size * rate / 2)
            )
            for x, y in coords
        ]

        expected = len(coords)
        existing = fast_count_png(patch_dir)

        if expected > 0 and existing >= expected:
            with open(done_flag, "w") as f:
                f.write(f"done_time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"existing_png={existing}\n")
                f.write(f"expected_coords={expected}\n")
                f.write("auto_mark_done=true\n")
            print(f"Already extracted ({existing}/{expected}), auto mark .done and skip")
            continue

        count.value = 0

        pool = Pool(
            processes=args.num_process,
            initializer=init_worker,
            initargs=(wsi_path, patch_dir, patch_size_real, cur_level)
        )

        for _ in pool.imap_unordered(process, coords, chunksize=50):
            pass

        pool.close()
        pool.join()

        final_existing = fast_count_png(patch_dir)
        with open(done_flag, "w") as f:
            f.write(f"done_time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"new_generated={int(count.value)}\n")
            f.write(f"final_existing_png={final_existing}\n")
            f.write(f"expected_coords={expected}\n")
            f.write(f"target_level={args.target_level}\n")
            f.write(f"patch_size={args.patch_size}\n")
            f.write(f"num_process={args.num_process}\n")

        print(
            f"Finished {wsi_name}, "
            f"NEW patches: {count.value}, total existing: {final_existing}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extract WSI patches based on precomputed tissue masks"
    )
    parser.add_argument("--wsi_root", required=True, type=str)
    parser.add_argument("--mask_root", required=True, type=str)
    parser.add_argument("--patch_root", required=True, type=str)

    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--target_level", default=20, type=int)
    parser.add_argument("--num_process", default=80, type=int)
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
    run(args)


if __name__ == "__main__":
    main()
