#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
write_feat_txt_dino.py

Convert selected patch-coordinate txt files into selected patch-feature txt/json files.

Workflow
--------
This script is the third step of the patch clustering pipeline:

1. featureClustering_dino.py
   - convert global DINO features into one txt per WSI
   - cluster patch features within each WSI

2. slide_select_byCluster.py
   - generate patch-selection txt files for each WSI and each iteration

3. write_feat_txt_dino.py   <-- current script
   - read the original per-WSI feature txt
   - read the selected patch-coordinate txt
   - retrieve the corresponding patch feature vectors
   - write one output file per selection txt

Input
-----
A) feature_txt_path
   Directory containing one feature txt per WSI, each line like:
       {"(x, y)": [f1, f2, ...]}

B) result_txt_path
   Directory containing selected patch-coordinate txt files, typically:
       <wsi>kmeans_cls_0.txt
       <wsi>kmeans_cls_1.txt
       ...

Output
------
write_txt_dir/
    <wsi>kmeans_cls_0.txt
    <wsi>kmeans_cls_1.txt
    ...

Each output file stores a JSON list of selected patch features.
The default expected number of selected patches is 500 (= 50 clusters × 10 patches/cluster).

Notes
-----
- The original naming and matching logic is intentionally preserved.
- Each worker processes one WSI at a time.
- Output format is kept compatible with the downstream PRISM workflow.
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write selected DINO patch features for each WSI based on selected patch coordinates."
    )
    parser.add_argument(
        "--feature_txt_path",
        type=str,
        required=True,
        help="Directory containing one original feature txt per WSI.",
    )
    parser.add_argument(
        "--result_txt_path",
        type=str,
        required=True,
        help="Directory containing selected patch-coordinate txt files.",
    )
    parser.add_argument(
        "--write_txt_dir",
        type=str,
        required=True,
        help="Output directory for selected patch-feature files.",
    )
    parser.add_argument(
        "--process_count",
        type=int,
        default=10,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=500,
        help="Expected number of selected patches per output file. Default: 500.",
    )
    return parser


def decode_feature_txt(feat_txt: str) -> Dict[str, List[float]]:
    """
    Decode one per-WSI feature txt into a dictionary:
        "(x, y)" -> feature vector

    Expected input format per line:
        {"(x, y)": [f1, f2, ...]}
    """
    feat_dict: Dict[str, List[float]] = {}

    with open(feat_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feat_dict.update(json.loads(line))

    return feat_dict


def normalize_coord_key(s: str) -> str:
    """
    Normalize coordinate strings read from selection txt.

    This is needed because coordinates may be wrapped by quotes when exported
    through pandas.DataFrame.to_csv().
    """
    return s.strip().strip('"').strip("'")


def load_data(
    txt_path: str,
    feat_data: Dict[str, List[float]],
    expected_count: int = 500,
) -> List[List[float]]:
    """
    Read one selected coordinate txt and fetch the corresponding features.

    The input txt is expected to contain tab-separated coordinate strings,
    arranged as:
        rows    -> clusters
        columns -> selected patches per cluster in one iteration

    Returns
    -------
    feat_list:
        A flat list of selected patch feature vectors.

    Raises
    ------
    AssertionError
        If the number of retrieved features does not match expected_count.
    """
    feat_list: List[List[float]] = []
    feat_get = feat_data.get

    with open(txt_path, "r") as f:
        for line in f.read().splitlines():
            if not line:
                continue
            for data_idx in line.split("\t"):
                key = normalize_coord_key(data_idx)
                value = feat_get(key, None)
                if value is not None:
                    feat_list.append(value)

    assert len(feat_list) == expected_count, (
        f"{txt_path}: got {len(feat_list)} selected features, "
        f"expected {expected_count}"
    )
    return feat_list


def discover_wsi_tasks(
    feature_txt_path: str,
    result_txt_path: str,
) -> List[Tuple[str, str, str]]:
    """
    Discover all WSI-level tasks.

    Logic preserved from the original script:
    - enumerate result files using '*_0.txt'
    - infer WSI prefix as the substring before 'kmeans'
    - pair each WSI prefix with its original feature txt

    Returns
    -------
    tasks:
        List of tuples:
            (txt_name, feat_path, result_txt_path)
    """
    cls_list = sorted(glob.glob(os.path.join(result_txt_path, "*_0.txt")))
    txt_names = []

    for p in cls_list:
        base = os.path.basename(p)
        kpos = base.find("kmeans")
        if kpos == -1:
            continue
        txt_name = base[:kpos]
        txt_names.append(txt_name)

    txt_names = sorted(set(txt_names))

    tasks = []
    for txt_name in txt_names:
        feat_path = os.path.join(feature_txt_path, txt_name + ".txt")
        if not os.path.exists(feat_path):
            print(f"[skip] missing feature txt: {feat_path}")
            continue
        tasks.append((txt_name, feat_path, result_txt_path))

    return tasks


def process_one_wsi(task: Tuple[str, str, str, str, int]) -> Tuple[str, int, int]:
    """
    Process one WSI:
    - load original feature txt once
    - iterate through all selection txt files of this WSI
    - write selected feature json lists

    Parameters
    ----------
    task:
        (txt_name, feat_path, result_txt_path, write_txt_dir, expected_count)

    Returns
    -------
    (txt_name, n_total, n_written)
    """
    txt_name, feat_path, result_txt_path, write_txt_dir, expected_count = task

    feat_data = decode_feature_txt(feat_path)

    # Keep original matching style
    select_paths = sorted(glob.glob(os.path.join(result_txt_path, txt_name + "*.txt")))
    if len(select_paths) == 0:
        return (txt_name, 0, 0)

    n_written = 0
    n_total = len(select_paths)

    for cls_path in select_paths:
        out_name = os.path.basename(cls_path)
        write_path = os.path.join(write_txt_dir, out_name)

        if os.path.exists(write_path):
            continue

        all_feat_list = load_data(
            txt_path=cls_path,
            feat_data=feat_data,
            expected_count=expected_count,
        )

        with open(write_path, "w") as f:
            json.dump(all_feat_list, f)

        n_written += 1

    return (txt_name, n_total, n_written)


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.write_txt_dir, exist_ok=True)

    tasks_base = discover_wsi_tasks(
        feature_txt_path=args.feature_txt_path,
        result_txt_path=args.result_txt_path,
    )

    tasks = [
        (txt_name, feat_path, result_txt_path, args.write_txt_dir, args.expected_count)
        for txt_name, feat_path, result_txt_path in tasks_base
    ]

    print(f"Total WSI tasks: {len(tasks)} | processes={args.process_count}")

    if len(tasks) == 0:
        print("[WARN] No valid WSI tasks found.")
        return

    total_written = 0
    total_target = 0

    with Pool(processes=args.process_count) as pool:
        for txt_name, n_total, n_written in tqdm(
            pool.imap_unordered(process_one_wsi, tasks),
            total=len(tasks),
        ):
            total_target += n_total
            total_written += n_written

    print(f"[DONE] total target files: {total_target}")
    print(f"[DONE] newly written files: {total_written}")
    print(f"[DONE] output dir: {args.write_txt_dir}")


if __name__ == "__main__":
    main()
