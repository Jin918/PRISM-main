#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
featureClustering_dino.py

Utilities for patch-level feature preparation and clustering based on DINO features.

This script provides two main workflows:

1) run1 / mode=cluster
   - Input: one .txt feature file per WSI
   - Operation: cluster patch features within each WSI (default: KMeans)
   - Output:
       * cluster assignment text file: *_cls.txt
       * spatial cluster visualization: *.png

2) run2 / mode=convert
   - Input:
       * patch feature tensor (.pth)
       * patch path list (.json)
   - Operation: convert global patch features into one .txt file per WSI
   - Output:
       * one .txt feature file per WSI

The implementation keeps the original data format and downstream compatibility.
"""

import argparse
import glob
import json
import os
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from skfuzzy.cluster import cmeans


RNG = np.random.RandomState(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch feature conversion and clustering for DINO-based WSI features."
    )

    # Keep original paths/argument names for compatibility
    parser.add_argument(
        "--feat_path",
        default="/Pathology_data/UCEC/ucec_all_dino_feat_txt",
        type=str,
        help=(
            "Path to the input feature source. "
            "For mode=cluster: directory containing per-WSI .txt files. "
            "For mode=convert: path to trainfeat.pth or equivalent tensor file."
        ),
    )
    parser.add_argument(
        "--position_path",
        default="/Pathology_data/UCEC/UCEC_full_features/ep10/train_paths.json",
        type=str,
        help="Path to patch path list JSON used in mode=convert.",
    )
    parser.add_argument(
        "--save_txt_path",
        default="/Pathology_data_2/UCEC/ucec_all_dino_feat_txt",
        type=str,
        help="Output directory for per-WSI .txt feature files in mode=convert.",
    )
    parser.add_argument(
        "--txt_path",
        default="/Pathology_data_2/UCEC/ucec_all_dino_feat_clustering",
        type=str,
        help="Output directory for clustering assignment text files in mode=cluster.",
    )
    parser.add_argument(
        "--png_path",
        default="/Pathology_data_2/UCEC/ucec_all_dino_feat_clustering",
        type=str,
        help="Output directory for clustering visualization PNG files in mode=cluster.",
    )
    parser.add_argument(
        "--class_num",
        default=50,
        type=int,
        help="Number of clustering classes.",
    )
    parser.add_argument(
        "--scale_ratio",
        default=512,
        type=int,
        help="Reserved for compatibility with previous versions. Not used in current script.",
    )
    parser.add_argument(
        "--chunk",
        default=4096,
        type=int,
        help="Buffered write chunk size for mode=convert.",
    )
    parser.add_argument(
        "--mode",
        default="cluster",
        choices=["cluster", "convert"],
        help=(
            "cluster: run original run1 workflow (default). "
            "convert: run original run2 workflow."
        ),
    )

    return parser


def randomcolor(class_num: int) -> List[str]:
    """Generate a reproducible list of random hex colors."""
    color_list = []
    color_arr = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"]
    for _ in range(class_num):
        color = "".join(color_arr[RNG.randint(0, 14)] for _ in range(6))
        color_list.append("#" + color)
    return color_list


def loadDataSet(file_name: str) -> Tuple[List[str], List[List[float]]]:
    """
    Load per-WSI feature txt file.

    Expected line format (kept compatible with original downstream scripts):
        {"(x, y)": [f1, f2, ...]}

    Returns
    -------
    feat_name:
        List of coordinate strings such as "(x, y)"
    feats:
        List of feature vectors
    """
    feat_name: List[str] = []
    feats: List[List[float]] = []

    with open(file_name, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            if len(data) != 1:
                raise ValueError(f"Unexpected line format in {file_name}: {line}")

            coord, feat = next(iter(data.items()))
            feat_name.append(coord)
            feats.append(feat)

    return feat_name, feats


def load_position(position_path: str) -> Tuple[List[int], List[str], List[str]]:
    """
    Load patch path list and infer:
    - WSI boundaries within the global feature tensor
    - WSI name for each patch
    - patch coordinate string for each patch

    Original naming assumption is preserved:
    - wsi name: parent directory name
    - patch basename without extension contains coordinates
    - coordinates are parsed from the substring after the first underscore
    """
    wsi_count: List[int] = []

    with open(position_path, "r") as f:
        path_data = json.load(f)

    wsi_name_list = [os.path.split(os.path.dirname(x))[-1] for x in path_data]
    patch_name_list = [os.path.basename(x)[:-4] for x in path_data]
    patch_position_list = [x[x.find("_") + 1:] for x in patch_name_list]

    if len(wsi_name_list) == 0:
        return [], [], []

    temp_wsi_name = wsi_name_list[0]
    for patch_idx, wsi_name in enumerate(wsi_name_list):
        if temp_wsi_name != wsi_name:
            wsi_count.append(patch_idx)
            temp_wsi_name = wsi_name
    wsi_count.append(len(wsi_name_list))

    return wsi_count, wsi_name_list, patch_position_list


def showClass(
    args: argparse.Namespace,
    json_name: str,
    feat_name: Sequence[str],
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    color_list: Sequence[str],
    ext: str = "",
) -> None:
    """
    Plot spatial cluster map for one WSI.

    Parameters
    ----------
    json_name:
        Input feature txt file name
    feat_name:
        Patch coordinate strings like "(x, y)"
    labels:
        Cluster label per patch
    ext:
        Suffix added before .png, e.g. 'kmeans'
    """
    del features  # kept in signature for compatibility with original function

    os.makedirs(args.png_path, exist_ok=True)
    png_file = os.path.join(args.png_path, json_name.replace(".txt", ext + ".png"))

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, coords in enumerate(feat_name):
        coord = coords.split(",")
        x = int(coord[0][1:])
        y = int(coord[1][1:-1])
        ax.plot(x, y, color=color_list[labels[i]], marker=".", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.savefig(png_file, bbox_inches="tight", dpi=1000)
    plt.close(fig)


def showEvaluate(
    args: argparse.Namespace,
    feat_name: Sequence[str],
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    json_name: str,
    ext: str,
) -> None:
    """
    Save cluster assignments as a two-column text file:
        coordinate<TAB>cluster_label
    """
    del features  # kept in signature for compatibility with original function

    os.makedirs(args.txt_path, exist_ok=True)

    data = list(zip(feat_name, labels))
    data = pd.DataFrame(data)
    txt_name = json_name.replace(".txt", ext + "_cls.txt")
    txt_path = os.path.join(args.txt_path, txt_name)
    data.to_csv(txt_path, sep="\t", index=False, header=False)


def kmeans_train(features: Sequence[Sequence[float]], clusters: int) -> KMeans:
    """
    KMeans clustering.

    The original script did not explicitly set random_state or n_init.
    This is intentionally kept minimal to avoid changing behavior.
    """
    return KMeans(n_clusters=clusters).fit(features)


def spectral_train(features: Sequence[Sequence[float]], clusters: int) -> SpectralClustering:
    """Spectral clustering helper retained for compatibility."""
    return SpectralClustering(n_clusters=clusters).fit(features)


def fcm_train(features: Sequence[Sequence[float]], clusters: int) -> np.ndarray:
    """Fuzzy C-means clustering helper retained for compatibility."""
    feature_t = np.array(features).T
    center, u, u0, d, jm, p, fpc = cmeans(
        feature_t, m=2, c=clusters, error=0.0001, maxiter=1000
    )
    del center, u0, d, jm, p, fpc
    return np.argmax(u, axis=0)


def pca_train(features: Sequence[Sequence[float]], cls_num: int) -> np.ndarray:
    """Optional PCA helper retained for compatibility."""
    pca = PCA(n_components=cls_num, svd_solver="arpack")
    new_feature = pca.fit_transform(features)
    return new_feature


def run1(args: argparse.Namespace) -> None:
    """
    Original clustering workflow.

    Input:
        args.feat_path -> directory with per-WSI .txt feature files
    Output:
        args.txt_path  -> *_cls.txt
        args.png_path  -> cluster map PNG
    """
    os.makedirs(args.txt_path, exist_ok=True)
    os.makedirs(args.png_path, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.feat_path, "*.txt")))
    color_list = randomcolor(args.class_num)

    for path in paths:
        print(path)
        json_name = os.path.basename(path)

        # Keep original "skip if PNG already exists" logic
        if os.path.exists(os.path.join(args.png_path, json_name.replace(".txt", "kmeans.png"))):
            continue

        feat_name, features = loadDataSet(path)
        if len(features) < 1:
            continue

        cls_nums = args.class_num if len(features) >= args.class_num else len(features)

        print("Clustering....")
        kmeans_cls = kmeans_train(features, cls_nums)

        print("Writing...")
        showEvaluate(args, feat_name, features, kmeans_cls.labels_, json_name, "kmeans")

        print("Ploting...")
        showClass(args, json_name, feat_name, features, kmeans_cls.labels_, color_list, "kmeans")


def run2(args: argparse.Namespace) -> None:
    """
    Convert global feature tensor + patch path list into one per-WSI .txt file.

    Input:
        args.feat_path     -> .pth tensor or tensor-like object
        args.position_path -> JSON list of patch file paths
    Output:
        args.save_txt_path -> one .txt per WSI
    """
    os.makedirs(args.save_txt_path, exist_ok=True)

    wsi_count_list, wsi_name_list, patch_position_list = load_position(args.position_path)
    if len(wsi_count_list) == 0:
        print(f"[WARN] No patch positions found in: {args.position_path}")
        return

    # Faster x/y parsing while preserving original coordinate semantics
    xs = np.empty(len(patch_position_list), dtype=np.int32)
    ys = np.empty(len(patch_position_list), dtype=np.int32)
    for i, s in enumerate(patch_position_list):
        a, b = s.split("_")
        xs[i] = int(a)
        ys[i] = int(b)

    patch_features = torch.load(args.feat_path, map_location="cpu")
    if isinstance(patch_features, torch.Tensor):
        patch_features = patch_features.cpu().numpy()
    patch_features = np.asarray(patch_features, dtype=np.float32)

    chunk = args.chunk
    start = 0

    for end in wsi_count_list:
        wsi = wsi_name_list[start]
        wsi_txt_path = os.path.join(args.save_txt_path, wsi + ".txt")

        print(f"[{wsi}] patches: {end - start} -> {wsi_txt_path}")

        # Buffered write to reduce system calls
        with open(wsi_txt_path, "w", buffering=32 * 1024 * 1024) as f:
            for s in range(start, end, chunk):
                e = min(end, s + chunk)

                lines = []
                for i in range(s, e):
                    coord = f"({int(xs[i])}, {int(ys[i])})"

                    # Keep ', ' separator for compatibility with historical format
                    feat_str = ", ".join(map(str, patch_features[i].tolist()))
                    lines.append(f'{{"{coord}": [{feat_str}]}}\n')

                f.write("".join(lines))

        start = end


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "cluster":
        run1(args)
    elif args.mode == "convert":
        run2(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
