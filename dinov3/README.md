# DINOv3 Adaptation and Feature Extraction

This directory contains the DINOv3-related scripts used in the PRISM pipeline for pathology-domain adaptation, patch-level feature extraction, and proxy evaluation.

These scripts are designed to work with the **official DINOv3 backbone** and were used to adapt natural-image pretrained representations to H&E pathology patches from TCGA-UCEC and CPTAC-UCEC. The resulting features are then used for downstream clustering, fixed-length WSI token construction, and PRISM multimodal survival modeling.

This module includes:
- in-domain self-supervised adaptation of DINOv3 on pathology patches
- patch-level feature extraction using official or adapted DINOv3 checkpoints
- patient-level linear-probe proxy evaluation
- trend plotting for checkpoint selection

---

## What is included in this folder

~~~text
dinov3/
├── README.md
├── main_dinov3.py
├── get_features_dinov3.py
├── eval_patient_probe.py
├── plot_dino_trend.py
├── utils.py
├── requirements.txt
└── requirements-dev.txt
~~~

---

## Scope of this module

This folder contains the **PRISM-side DINOv3 workflow scripts**, not the full official DINOv3 source code.

The scripts depend on the official DINOv3 package/API, for example:

- `from dinov3.hub import backbones`
- `from dinov3.layers.dino_head import DINOHead`

Therefore, running this module requires that the official DINOv3 codebase or package is available in the Python environment.

In addition, the official pretrained weight used in this project was:

- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`

This weight file is treated as an external model asset rather than core source code.

---

## Workflow

The DINOv3 workflow in PRISM consists of four steps:

1. **Pathology-domain adaptation**  
   Adapt the official DINOv3 backbone on H&E patch images using teacher–student self-supervised learning.

2. **Patch-level feature extraction**  
   Extract one feature vector per patch using either:
   - the official pretrained model, or
   - an adapted checkpoint from pathology-domain training.

3. **Patient-level proxy evaluation**  
   Aggregate patch-level features to patient-level features and evaluate proxy prediction tasks such as histologic grade, molecular subtype, and binary clinical endpoints.

4. **Checkpoint trend visualization**  
   Compare proxy performance across adaptation epochs to support checkpoint selection.

---

## Files

### `main_dinov3.py`
Self-supervised in-domain adaptation of the official DINOv3 backbone on pathology patches.

This script implements a DINO-style teacher–student training framework for pathology-domain adaptation and supports both:

- **full finetuning**
- **lightweight finetuning**  
  (for example, training only the DINO head, last transformer blocks, and normalization layers)

**Main features**
- official DINOv3 backbone loading
- teacher–student self-distillation
- multi-crop augmentation
- full or lightweight finetuning policy
- distributed multi-GPU training
- resumable checkpointing
- per-epoch logging

**Typical outputs**
- `checkpoint.pth`
- `checkpoint0000.pth`, `checkpoint0001.pth`, ...
- `log.txt`

---

### `get_features_dinov3.py`
Patch-level feature extraction using the official DINOv3 backbone.

This script supports two modes:

- **baseline extraction** using official pretrained DINOv3 weights
- **adapted extraction** using a finetuned pathology-domain checkpoint

Each patch is transformed with a fixed inference pipeline and passed through the backbone to generate one embedding vector.

**Main features**
- official DINOv3 backbone construction
- load from official pretrained weights or user checkpoint
- distributed feature extraction
- automatic path-feature alignment
- L2 normalization of extracted features

**Typical outputs**
- `trainfeat.pth`
- `train_paths.json`
- `meta.json`

---

### `eval_patient_probe.py`
Patient-level proxy evaluation from extracted patch features.

This script aggregates patch-level embeddings to patient-level representations by mean pooling and evaluates whether the resulting features capture clinically relevant or biologically meaningful information.

Supported evaluation modes:
- **holdout**: train/validation split
- **cv**: stratified cross-validation

Supported task types:
- **multiclass**  
  such as `Histologic_Grade`, `Subtype`
- **binary**  
  such as `y12`, `y24`, `y36`, `y60`, `mi_50`, `Radiation_Therapy`, `Residual_Tumor`

**Main features**
- patch-to-patient aggregation
- linear probe with logistic regression
- binary and multiclass evaluation
- train/validation or cross-validation mode
- exportable CSV results

**Typical outputs**
- `patient_feats.npz`
- `eval_results_holdout_train_val.csv`
- `eval_results_cv.csv`

---

### `plot_dino_trend.py`
Trend plotting for DINOv3 adaptation checkpoints.

This script reads the proxy evaluation results from multiple runs and summarizes how validation performance changes across adaptation epochs.

It is mainly used to support checkpoint selection during pathology-domain adaptation.

**Main features**
- parse multiple runs from `name=csv_path`
- infer epoch from run name
- summarize binary task AUC trends
- summarize multiclass balanced-accuracy trends
- generate publication-ready checkpoint trend plots

**Typical outputs**
- `trend_metrics.csv`
- `trend_auc.png`
- `trend_balacc.png`

---

### `utils.py`
Utility functions used across the DINOv3 adaptation and extraction scripts.

This file contains:
- distributed training initialization
- checkpoint restart helpers
- cosine learning-rate scheduling
- logging utilities
- multi-crop wrapper
- DINO-style data augmentation helpers
- gradient clipping and optimizer helpers

This file is a project-adapted utility layer derived from public DINO-style implementations and used to keep the training and feature extraction workflow self-contained.

---

## Inputs

### Required data
- pathology patch images organized in `ImageFolder` format
- optional train/validation clinical CSV files for proxy evaluation
- optional pretrained checkpoint for adapted feature extraction

### Patch data format
The scripts assume the patch images are organized in a standard folder structure compatible with `torchvision.datasets.ImageFolder`, for example:

~~~text
patch_data_whole_slide_20X_256/
├── TCGA-XX-XXXX-01Z-00-DX1/
│   ├── patch_000001.png
│   ├── patch_000002.png
│   └── ...
├── TCGA-YY-YYYY-01Z-00-DX1/
│   ├── patch_000001.png
│   └── ...
~~~

Patch paths are later used to recover patient identifiers for patient-level aggregation.

---

## Dependencies

### Python packages
Core dependencies include:
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `Pillow`

### Additional requirement
The official DINOv3 backbone code must be available in the environment, since these scripts import:

- `dinov3.hub.backbones`
- `dinov3.layers.dino_head`
- optional `dinov3.layers.rms_norm`

---

## Example Usage

### 1. Pathology-domain adaptation

#### Full finetuning
~~~bash
conda activate dinov3_py311

torchrun --nproc_per_node=8 main_dinov3.py \
  --arch dinov3_vits16plus \
  --pretrained_weights /root/code/PRISM-main/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth \
  --data_path /Pathology_data/UCEC/patch_data_whole_slide_20X_256 \
  --output_dir /Pathology_data/UCEC/log_UCEC_dinov3_full \
  --epochs 20 \
  --saveckp_freq 1 \
  --batch_size_per_gpu 128 \
  --num_workers 8 \
  --use_fp16 true \
  --warmup_epochs 1 \
  --warmup_teacher_temp_epochs 3 \
  --lr 0.0002
~~~

#### Lightweight finetuning
~~~bash
conda activate dinov3_py311

torchrun --nproc_per_node=8 main_dinov3.py \
  --arch dinov3_vits16plus \
  --pretrained_weights /root/code/PRISM-main/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth \
  --data_path /Pathology_data/UCEC/patch_data_whole_slide_20X_256 \
  --output_dir /Pathology_data/UCEC/log_UCEC_dinov3_light \
  --epochs 9 \
  --saveckp_freq 1 \
  --batch_size_per_gpu 128 \
  --num_workers 8 \
  --use_fp16 true \
  --warmup_epochs 1 \
  --warmup_teacher_temp_epochs 3 \
  --lightweight_finetune true \
  --train_last_n_blocks 2 \
  --train_norm_layers true \
  --lr 0.0002
~~~

---

### 2. Feature extraction

#### Baseline extraction from official pretrained weights
~~~bash
conda activate dinov3_py311

torchrun --nproc_per_node=8 get_features_dinov3.py \
  --arch dinov3_vits16plus \
  --weights /root/code/PRISM-main/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth \
  --data_path /Pathology_data/UCEC/patch_data_whole_slide_20X_256 \
  --dump_features /Pathology_data/UCEC/UCEC_features_dinov3_pretrain \
  --batch_size_per_gpu 256 \
  --num_workers 10 \
  --amp true
~~~

#### Feature extraction from a finetuned checkpoint
~~~bash
conda activate dinov3_py311

CKPT=/Pathology_data/UCEC/log_UCEC_dinov3_full/checkpoint0010.pth
DATA=/Pathology_data_2/UCEC_new/CPTAC/patch_data_whole_slide_20X_256
OUT=/Pathology_data_2/UCEC_new/CPTAC/wsi_features_dinov3_full_ep10

torchrun --nproc_per_node=8 get_features_dinov3.py \
  --arch dinov3_vits16plus \
  --checkpoint_pth "$CKPT" \
  --checkpoint_key teacher \
  --data_path "$DATA" \
  --dump_features "$OUT" \
  --batch_size_per_gpu 256 \
  --num_workers 10 \
  --amp true
~~~

---

### 3. Patient-level proxy evaluation

#### Directly from extracted patch features
~~~bash
conda activate dinov3_py311

python eval_patient_probe.py \
  --feat_pth /Pathology_data/UCEC/UCEC_features_dinov3_light_ep8/trainfeat.pth \
  --paths_json /Pathology_data/UCEC/UCEC_features_dinov3_light_ep8/train_paths.json \
  --save_patient_npz /Pathology_data/UCEC/UCEC_features_dinov3_light_ep8/patient_feats.npz \
  --mode holdout \
  --train_csv /root/code/PRISM-main/datasets/train_vali/split_287/EC_Clinic_PFS_287.train80.csv \
  --val_csv /root/code/PRISM-main/datasets/train_vali/split_287/EC_Clinic_PFS_287.val20.csv \
  --out_dir /Pathology_data/UCEC/UCEC_features_dinov3_light_ep8/eval_out
~~~

#### Reuse precomputed patient-level features
~~~bash
conda activate dinov3_py311

python eval_patient_probe.py \
  --patient_npz /Pathology_data/UCEC/UCEC_features_dinov3_light_ep8/patient_feats.npz \
  --mode holdout \
  --train_csv /root/code/PRISM-main/datasets/train_vali/split_287/EC_Clinic_PFS_287.train80.csv \
  --val_csv /root/code/PRISM-main/datasets/train_vali/split_287/EC_Clinic_PFS_287.val20.csv \
  --out_dir /Pathology_data/UCEC/UCEC_features_dinov3_light_ep8/eval_out
~~~

---

### 4. Plot checkpoint trends

~~~bash
conda activate dinov3_py311

python plot_dino_trend.py \
  --runs baseline=/Pathology_data/UCEC/UCEC_features_dinov3_baseline/eval_out/eval_results_holdout_train_val.csv \
        ep4=/Pathology_data/UCEC/UCEC_features_dinov3_light_ep4/eval_out/eval_results_holdout_train_val.csv \
        ep6=/Pathology_data/UCEC/UCEC_features_dinov3_light_ep6/eval_out/eval_results_holdout_train_val.csv \
        ep8=/Pathology_data/UCEC/UCEC_features_dinov3_light_ep8/eval_out/eval_results_holdout_train_val.csv \
  --out_dir /Pathology_data/UCEC/dinov3_trend_plots
~~~

---

## Key Outputs and Their Meanings

### Adaptation outputs
- `checkpoint.pth`  
  latest training checkpoint
- `checkpointXXXX.pth`  
  per-epoch checkpoint snapshots
- `log.txt`  
  epoch-wise training log

### Feature extraction outputs
- `trainfeat.pth`  
  patch-level feature matrix
- `train_paths.json`  
  patch paths aligned row-by-row with `trainfeat.pth`
- `meta.json`  
  extraction metadata

### Patient-level evaluation outputs
- `patient_feats.npz`  
  mean-pooled patient-level features
- `eval_results_holdout_train_val.csv`  
  holdout proxy evaluation results
- `eval_results_cv.csv`  
  cross-validation proxy evaluation results

### Trend plotting outputs
- `trend_metrics.csv`
- `trend_auc.png`
- `trend_balacc.png`

---

## How this module connects to PRISM

The outputs of this directory are used in the downstream PRISM workflow as follows:

1. pathology patches are extracted from WSIs
2. DINOv3 is adapted on pathology patches
3. patch-level embeddings are extracted
4. embeddings are clustered and sampled to build fixed-length WSI tokens
5. WSI tokens are used together with pathway/program tokens in PRISM multimodal survival modeling

Thus, this module provides the pathology representation backbone for the full PRISM pipeline.

---

## Reproducibility Notes

- The scripts support distributed multi-GPU execution with `torchrun`.
- Existing checkpoints can be resumed automatically during adaptation.
- Feature extraction stores both features and aligned patch paths to ensure reproducible aggregation.
- Patient-level evaluation uses deterministic aggregation from patch to patient.
- Checkpoint selection can be guided by proxy tasks such as histologic grade and molecular subtype prediction.

---

## Summary

This directory provides the DINOv3-based pathology representation workflow used in PRISM:

- `main_dinov3.py`  
  pathology-domain self-supervised adaptation

- `get_features_dinov3.py`  
  patch-level feature extraction

- `eval_patient_probe.py`  
  patient-level proxy evaluation

- `plot_dino_trend.py`  
  checkpoint trend visualization

- `utils.py`  
  shared utilities for distributed training and extraction

Together, these scripts adapt the official DINOv3 backbone to pathology images and generate patch-level representations for downstream interpretable multimodal survival analysis.
