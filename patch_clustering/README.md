# PRISM patch clustering and patch selection

This directory contains the DINO-feature-based patch clustering pipeline used in PRISM for WSI patch subsampling. The pipeline converts raw patch feature tensors into per-WSI feature files, performs within-slide clustering, selects representative patches from each cluster, and writes the selected patch features for downstream survival modeling.

The current implementation is designed to preserve the original project workflow while improving code organization, readability, and reproducibility.

---

## Overview

This module implements a three-step pipeline:

1. **feature conversion and clustering**
   - convert global patch feature tensors into one feature `.txt` file per WSI
   - cluster patch-level DINO features within each WSI
   - export cluster assignment files and spatial visualization PNGs

2. **cluster-guided patch selection**
   - group patch coordinates by cluster
   - sample a fixed number of patches from each cluster
   - generate one or more patch-selection `.txt` files per WSI

3. **selected feature writing**
   - map selected patch coordinates back to the original feature dictionary
   - export selected patch feature sets for downstream PRISM modeling

---

## Directory contents

| File | Description |
|---|---|
| `featureClustering_dino.py` | Converts tensor features to per-WSI txt files and performs within-slide clustering |
| `slide_select_byCluster.py` | Selects representative patch coordinates from clustering results |
| `write_feat_txt_dino.py` | Writes selected patch feature lists for downstream modeling |

---

## Expected workflow

The intended execution order is:

1. prepare DINO patch features and patch-path JSON
2. run `featureClustering_dino.py` in `convert` mode
3. run `featureClustering_dino.py` in `cluster` mode
4. run `slide_select_byCluster.py`
5. run `write_feat_txt_dino.py`
6. use the exported selected patch features as WSI inputs for downstream PRISM survival modeling

---

## Input and output structure

### Step 1. Feature conversion

#### Inputs
- a global patch feature tensor, for example:
  - `trainfeat.pth`
- a patch path list JSON, for example:
  - `train_paths.json`

#### Output
One `.txt` file per WSI:

```text
ucec_all_dino_feat_txt/
├── TCGA-XX-XXXX.txt
├── TCGA-XX-YYYY.txt
└── ...
```

Each line in a per-WSI feature file has the format:

```text
{"(x, y)": [f1, f2, f3, ...]}
```

---

### Step 2. Patch clustering

#### Input
Per-WSI feature `.txt` files generated in Step 1.

#### Outputs
- cluster assignment files:
  - `*_kmeans_cls.txt`
- cluster visualization PNGs:
  - `*kmeans.png`

Typical output layout:

```text
ucec_all_dino_feat_clustering/
├── TCGA-XX-XXXXkmeans_cls.txt
├── TCGA-XX-XXXXkmeans.png
├── TCGA-XX-YYYYkmeans_cls.txt
├── TCGA-XX-YYYYkmeans.png
└── ...
```

Each clustering assignment line is:

```text
(x, y)    cluster_id
```

---

### Step 3. Patch selection by cluster

#### Input
Cluster assignment files from Step 2.

#### Output
One or more patch-selection `.txt` files per WSI:

```text
ucec_whole_slide_select_txt_dino/
├── TCGA-XX-XXXXkmeans_cls_0.txt
├── TCGA-XX-XXXXkmeans_cls_1.txt
├── TCGA-XX-YYYYkmeans_cls_0.txt
└── ...
```

Each output file stores selected patch coordinates in tab-separated format, organized by cluster.

By default:
- number of clusters = `50`
- number of selected patches per cluster per round = `10`

Therefore, each selection file typically corresponds to `50 × 10 = 500` selected patches.

---

### Step 4. Selected feature writing

#### Inputs
- original per-WSI feature `.txt` files from Step 1
- patch-selection `.txt` files from Step 3

#### Output
One JSON list per patch-selection file:

```text
ucec_whole_slide_select_feat_txt_dino/
├── TCGA-XX-XXXXkmeans_cls_0.txt
├── TCGA-XX-XXXXkmeans_cls_1.txt
├── TCGA-XX-YYYYkmeans_cls_0.txt
└── ...
```

Each output file contains a JSON list of selected patch feature vectors.

These files are used as downstream WSI feature inputs for PRISM training and inference.

---

## Script details

### 1. `featureClustering_dino.py`

This script supports two modes.

#### Mode `convert`
Converts a global patch tensor plus patch path JSON into one per-WSI feature `.txt` file.

#### Mode `cluster`
Performs clustering within each WSI and exports:
- cluster assignment `.txt`
- spatial cluster map `.png`

#### Main arguments

| Argument | Description |
|---|---|
| `--mode` | `convert` or `cluster` |
| `--feat_path` | Tensor `.pth` file for `convert`, or per-WSI feature txt directory for `cluster` |
| `--position_path` | Patch path JSON for `convert` |
| `--save_txt_path` | Output directory for per-WSI feature txt files |
| `--txt_path` | Output directory for clustering assignment txt files |
| `--png_path` | Output directory for clustering PNGs |
| `--class_num` | Number of clusters |
| `--chunk` | Buffered write chunk size for `convert` |

#### Example: convert mode

```bash
mkdir -p /Pathology_data_2/UCEC_external/ucec_all_dino_feat_txt_light_8

python featureClustering_dino.py \
  --mode convert \
  --feat_path /Pathology_data_2/UCEC_external/UCEC_lingt_features_ep8/trainfeat.pth \
  --position_path /Pathology_data_2/UCEC_external/UCEC_lingt_features_ep8/train_paths.json \
  --save_txt_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_txt_light_8 \
  --chunk 8192
```

#### Example: cluster mode

```bash
mkdir -p /Pathology_data_2/UCEC_external/ucec_all_dino_feat_clustering_light_8

python featureClustering_dino.py \
  --mode cluster \
  --feat_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_txt_light_8 \
  --txt_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_clustering_light_8 \
  --png_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_clustering_light_8 \
  --class_num 50
```

---

### 2. `slide_select_byCluster.py`

This script selects representative patch coordinates from clustering results.

For each WSI:
- patches are grouped by cluster
- coordinates are shuffled within each cluster
- the maximum cluster size determines how many selection rounds are needed
- if a cluster is too small, coordinates are repeated to preserve balanced sampling
- one output `.txt` is written for each selection round

#### Main arguments

| Argument | Description |
|---|---|
| `--txt_path` | Directory containing clustering assignment txt files |
| `--txt_res_path` | Output directory for selected patch coordinate txt files |
| `--cls_num` | Number of clusters |
| `--num_per_cls` | Number of patches selected per cluster in each round |

#### Example

```bash
mkdir -p /Pathology_data_2/UCEC_new/TCGA/ucec_whole_slide_select_txt_dino_full_ep10_ge0p1

python slide_select_byCluster.py \
  --txt_path /Pathology_data_2/UCEC_new/TCGA/ucec_all_dino_feat_clustering_full_ep10_ge0p1 \
  --txt_res_path /Pathology_data_2/UCEC_new/TCGA/ucec_whole_slide_select_txt_dino_full_ep10_ge0p1 \
  --cls_num 50 \
  --num_per_cls 10
```

---

### 3. `write_feat_txt_dino.py`

This script maps selected patch coordinates back to the original feature dictionary and writes selected patch feature lists.

Each worker processes one WSI:
- load original per-WSI feature dictionary
- find all patch-selection files for that WSI
- retrieve the selected features by coordinate
- write one JSON list for each selection file

#### Main arguments

| Argument | Description |
|---|---|
| `--feature_txt_path` | Directory containing original per-WSI feature txt files |
| `--result_txt_path` | Directory containing selected patch coordinate txt files |
| `--write_txt_dir` | Output directory for selected patch feature files |
| `--process_count` | Number of worker processes |
| `--expected_count` | Expected number of selected patches per output file |

#### Example

```bash
mkdir -p /Pathology_data_2/UCEC_new/TCGA/ucec_whole_slide_select_feat_txt_dino_full_ep10_ge0p1

python write_feat_txt_dino.py \
  --feature_txt_path /Pathology_data_2/UCEC_new/TCGA/ucec_all_dino_feat_txt_full_ep10_ge0p1 \
  --result_txt_path /Pathology_data_2/UCEC_new/TCGA/ucec_whole_slide_select_txt_dino_full_ep10_ge0p1 \
  --write_txt_dir /Pathology_data_2/UCEC_new/TCGA/ucec_whole_slide_select_feat_txt_dino_full_ep10_ge0p1 \
  --process_count 10
```

---

## Typical end-to-end example

```bash
# Step 1: convert tensor features to per-WSI txt
mkdir -p /Pathology_data_2/UCEC_external/ucec_all_dino_feat_txt_light_8

python featureClustering_dino.py \
  --mode convert \
  --feat_path /Pathology_data_2/UCEC_external/UCEC_lingt_features_ep8/trainfeat.pth \
  --position_path /Pathology_data_2/UCEC_external/UCEC_lingt_features_ep8/train_paths.json \
  --save_txt_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_txt_light_8 \
  --chunk 8192

# Step 2: cluster patches within each WSI
mkdir -p /Pathology_data_2/UCEC_external/ucec_all_dino_feat_clustering_light_8

python featureClustering_dino.py \
  --mode cluster \
  --feat_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_txt_light_8 \
  --txt_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_clustering_light_8 \
  --png_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_clustering_light_8 \
  --class_num 50

# Step 3: select representative coordinates from clusters
mkdir -p /Pathology_data_2/UCEC_external/ucec_whole_slide_select_txt_dino_light_8

python slide_select_byCluster.py \
  --txt_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_clustering_light_8 \
  --txt_res_path /Pathology_data_2/UCEC_external/ucec_whole_slide_select_txt_dino_light_8 \
  --cls_num 50 \
  --num_per_cls 10

# Step 4: write selected patch features
mkdir -p /Pathology_data_2/UCEC_external/ucec_whole_slide_select_feat_txt_dino_light_8

python write_feat_txt_dino.py \
  --feature_txt_path /Pathology_data_2/UCEC_external/ucec_all_dino_feat_txt_light_8 \
  --result_txt_path /Pathology_data_2/UCEC_external/ucec_whole_slide_select_txt_dino_light_8 \
  --write_txt_dir /Pathology_data_2/UCEC_external/ucec_whole_slide_select_feat_txt_dino_light_8 \
  --process_count 10
```

---

## Dependencies

This directory requires the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `scikit-learn`
- `scikit-fuzzy`
- `tqdm`
- `openslide-python` (only needed if using optional image-cutting utilities)

A typical environment should also have OpenSlide system libraries installed if patch image extraction is needed.

---

## Notes

- The pipeline preserves the original project assumption that patch coordinates uniquely identify each feature vector within a WSI.
- The default setting assumes `50` clusters and `10` selected patches per cluster, leading to `500` selected patch features per output file.
- Cluster IDs are expected to span the full range from `0` to `cls_num - 1`.
- Selection is performed independently within each WSI.
- These scripts are intended for precomputed DINO patch features and do not perform feature extraction from raw images.

---

## Reproducibility

The scripts use deterministic random seeds where applicable for:
- color generation
- cluster-wise shuffling during patch selection

For full reproducibility, it is recommended to record:
- input feature directories
- patch path JSON files
- cluster count
- selected patch count per cluster
- process count
- output directory names used for each run

---

## Suggested citation context

This directory corresponds to the patch clustering and cluster-guided patch subsampling component of PRISM. It was used to generate:

- per-WSI DINO patch feature txt files
- within-slide clustering assignments
- representative cluster-balanced patch selections
- downstream selected WSI feature inputs for PRISM survival modeling
