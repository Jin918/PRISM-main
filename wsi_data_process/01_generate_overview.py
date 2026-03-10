#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from pathlib import Path

import openslide
import numpy as np
import imageio.v2 as imageio


def extract_case_id(s: str):
    """
    支持：
    - CPTAC: C3L-00001 / C3N-00001
    - TCGA : TCGA-XX-XXXX
    """
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


def load_clinical_ids(path: str):
    ids = set()
    if path and os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    ids.add(s.upper())
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


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clin_ids = load_clinical_ids(args.clinical_ids)
    if args.clinical_ids and not clin_ids:
        print(f"[WARN] clinical_ids is empty or not found: {args.clinical_ids}")
        print("[WARN] will process all matched WSI files.")

    paths = collect_wsi_paths(args.wsi_path, args.suffix)

    if clin_ids:
        kept = []
        for p in paths:
            cid = extract_case_id(os.path.basename(p)) or extract_case_id(p)
            if cid is not None and cid in clin_ids:
                kept.append(p)
        paths = kept

    print(f"[INFO] matched svs count = {len(paths)}")

    ok = 0
    skip = 0
    fail = 0

    for path in paths:
        name = os.path.basename(path)
        out_jpg = out_dir / f"{Path(name).stem}.jpg"

        if out_jpg.exists():
            skip += 1
            continue

        slide = None
        try:
            slide = openslide.OpenSlide(path)
            lvl = min(args.level, slide.level_count - 1)

            img_rgb = np.transpose(
                np.array(
                    slide.read_region(
                        (0, 0),
                        lvl,
                        slide.level_dimensions[lvl]
                    ).convert("RGB")
                ),
                axes=[1, 0, 2]
            )
            slide.close()
            slide = None

            imageio.imwrite(str(out_jpg), img_rgb)
            ok += 1

        except Exception as e:
            print(f"[SKIP] {path} failed: {e}")
            fail += 1
            try:
                if slide is not None:
                    slide.close()
            except Exception:
                pass

    print(f"[DONE] ok={ok}, skip={skip}, fail={fail}")


def main():
    parser = argparse.ArgumentParser(
        description="Save WSI overview jpg at a given level"
    )
    parser.add_argument("--wsi_path", required=True, type=str)
    parser.add_argument("--out_dir", "--npy_path", dest="out_dir", required=True, type=str)
    parser.add_argument("--level", default=1, type=int)
    parser.add_argument(
        "--suffix",
        default=".svs",
        type=str,
        help="e.g. '.svs' or '-21.svs'; empty string means all .svs"
    )
    parser.add_argument(
        "--clinical_ids",
        default="",
        type=str,
        help="optional txt file, one case_id per line"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
