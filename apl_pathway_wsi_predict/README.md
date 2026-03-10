# PRISM survival modeling

This directory contains the core survival modeling code for **PRISM**, a multimodal framework for prognosis prediction from whole-slide image (WSI) features and pathway-gene features. It includes model training, internal validation, unimodal ablations, checkpoint export, and post hoc prediction-time interpretability utilities.

The current implementation is organized around a **fixed training/validation split**, which is the recommended manuscript-facing workflow for model development and internal model selection.

---

## Overview

This module implements:

- multimodal survival modeling with WSI and pathway-gene representations
- WSI-only and gene-only ablation models
- Cox-based optimization and evaluation
- time-dependent ROC computation during validation
- best-epoch prediction export for downstream survival analyses
- attention and prototype-assignment export for interpretability

This directory assumes that all upstream processing steps have already been completed, including:

- clinical and transcriptomic preprocessing
- generation of Cox survival text files
- pathway-gene feature construction
- WSI patch extraction
- DINOv3 feature extraction

---

## Directory contents

| File | Description |
|---|---|
| `train_survival.py` | Main training script for multimodal or unimodal survival modeling |
| `predict_survival.py` | Prediction and interpretability export script |
| `utils_cox.py` | Training/validation utilities for Cox loss, c-index, log-rank test, and time-dependent ROC |
| `utils_cox_predict.py` | Prediction-time utilities for inference and attention export |
| `wsi_gene_dataset.py` | Dataset loader for WSI features, pathway-gene features, and survival labels |
| `vit_model_gene_wsi_concat.py` | Main multimodal WSI + gene survival model |
| `vit_model_gene_wsi_concat_no_contrastive_loss.py` | Multimodal ablation model without contrastive supervision |
| `vit_model_gene_wsi_concat_predict.py` | Prediction-time multimodal model with attention export |
| `vit_model_one_cls.py` | WSI-only survival model |
| `gene_only.py` | Gene-only survival model |
| `apl_bottleneck.py` | APL prototype bottleneck and associated regularization terms |

---

## Model variants

The code supports three model modes through `train_flag`:

- `0`: multimodal WSI + gene model
- `1`: WSI-only model
- `2`: gene-only model

For the multimodal model (`train_flag=0`), `contrastive_loss_flag` controls whether the contrastive branch is enabled:

- `1`: with contrastive supervision
- `0`: without contrastive supervision

Optional prototype compression is controlled by `proto_k`:

- `proto_k > 0`: APL bottleneck enabled
- `proto_k <= 0`: APL bottleneck disabled

---

## Input requirements

This directory expects analysis-ready inputs generated upstream.

### 1. Survival files

Cox-format text files with three tab-separated columns:

```text
sample_id    time    event
```

Where:

- `sample_id` is the case identifier
- `time` is survival time
- `event` is the event indicator (`1` = event, `0` = censored)

Typical files:

- `train.txt`
- `val.txt`

### 2. WSI feature directories

Directories containing one precomputed WSI feature file per case.

Typical layout:

```text
wsi/train/
├── TCGA-XX-XXXX.txt
├── TCGA-XX-YYYY.txt
└── ...

wsi/val/
├── TCGA-XX-ZZZZ.txt
└── ...
```

### 3. Pathway-gene feature directories

Directories containing one pathway-gene matrix per case.

Typical layout:

```text
gene/train/
├── TCGA-XX-XXXX.npy
├── TCGA-XX-YYYY.npy
└── ...

gene/val/
├── TCGA-XX-ZZZZ.npy
└── ...
```

---

## Typical workflow

The intended manuscript-facing order is:

1. preprocess clinical and transcriptomic data
2. generate train/validation split
3. construct Cox survival files and pathway-gene matrices
4. extract WSI patches
5. extract DINOv3 patch features
6. train survival model with `train_survival.py`
7. export predictions and interpretability outputs with `predict_survival.py`
8. perform downstream survival statistics and figure generation in separate analysis scripts

---

## Training

### Main script

`train_survival.py` is the primary entry point for model training and internal validation.

It performs:

- dataset loading
- model construction
- optimizer and scheduler setup
- epoch-wise training
- validation-set monitoring
- best-checkpoint saving
- best-epoch prediction export

### Example command

```bash
python train_survival.py \
  --num_classes 1 \
  --epochs 18 \
  --topK 2 \
  --batch_size 128 \
  --lr 5e-4 \
  --lrf 3e-4 \
  --weight_decay 1e-4 \
  --train_flag 0 \
  --contrastive_loss_flag 1 \
  --seed 42 \
  --view_seed 42 \
  --cox_train_path /path/to/cox/train.txt \
  --cox_val_path /path/to/cox/val.txt \
  --wsi_train_feat_dir /path/to/wsi/train \
  --wsi_valid_feat_dir /path/to/wsi/val \
  --gene_train_feat_dir /path/to/gene/train \
  --gene_valid_feat_dir /path/to/gene/val \
  --proto_k 4 \
  --proto_tau 0.1 \
  --lambda_div 0.0 \
  --lambda_bal 0.1 \
  --wsi_block 2 \
  --gene_block 1 \
  --dpr 0.3 \
  --roc_years 1 3 5 \
  --days_per_year 12 \
  --roc_every 1 \
  --save_epoch_ckpt \
  --save_epoch_weights \
  --save_every 1 \
  --log_dir /path/to/output/run_name
```

### Key outputs from training

A typical run directory contains:

```text
run_name/
├── model_log.txt
├── train_valid_acc.txt
├── valid_log.txt
├── train_eval_log.txt
├── model-latest.pth
├── model-val-best.pth
├── model-sum-best.pth
├── ckpt-latest.pt
├── ckpt-val-best.pt
├── ckpt-sum-best.pt
├── pred_train_best.tsv
├── pred_valid_best.tsv
├── best_val_epoch.txt
├── tdroc_epoch001.png
├── ckpt-epoch-000.pt
├── model-epoch-000.pth
└── ...
```

---

## Prediction export

At the best validation epoch, the training script exports:

- `pred_train_best.tsv`
- `pred_valid_best.tsv`

Each file contains:

```text
sid    time    event    risk
```

These files are intended for downstream Kaplan-Meier, time-dependent ROC, calibration, and Cox regression analyses.

---

## Prediction and interpretability export

### Main script

`predict_survival.py` is used after training to load a selected checkpoint and run inference on a target set.

It supports:

- prediction on internal validation or external datasets
- ID normalization for feature-file matching
- canonical symlink-based feature staging
- attention export
- APL assignment export

### Example command

```bash
python predict_survival.py \
  --num_classes 1 \
  --topK 2 \
  --batch_size 128 \
  --train_flag 0 \
  --contrastive_loss_flag 1 \
  --cox_txt_path /path/to/cox/target.txt \
  --wsi_valid_feat_dir /path/to/wsi/target \
  --gene_valid_feat_dir /path/to/gene/target \
  --weights /path/to/model-val-best.pth \
  --wsi_block 2 \
  --gene_block 1 \
  --dpr 0.3 \
  --proto_k 4 \
  --proto_tau 0.1 \
  --log_dir /path/to/predict_output \
  --save_attn_dir /path/to/predict_output/attn
```

### Prediction outputs

A typical prediction directory contains:

```text
predict_output/
├── cox_norm.txt
├── valid_log.txt
├── model_log.txt
├── features_link/
│   ├── wsi/
│   └── gene/
└── attn/
    ├── head0/
    ├── head1/
    ├── ...
    └── head15/
```

For multimodal prediction with attention export:

- attention maps are saved as per-head `.pth` files
- APL assignment matrices are also saved when `proto_k > 0`

---

## Architecture notes

### Multimodal model

`vit_model_gene_wsi_concat.py` implements the core PRISM multimodal survival architecture:

- WSI token encoder
- pathway-gene token encoder
- gene-guided cross-attention fusion
- survival prediction head
- optional APL bottleneck for WSI token compression

### No-contrastive ablation

`vit_model_gene_wsi_concat_no_contrastive_loss.py` provides a strict ablation model that preserves the main multimodal structure while disabling contrastive supervision.

### Prediction-time model

`vit_model_gene_wsi_concat_predict.py` mirrors the multimodal model but additionally returns:

- cross-attention maps
- APL assignment matrices

for interpretability analyses.

### APL bottleneck

`apl_bottleneck.py` implements:

- learnable prototype compression
- prototype diversity regularization
- assignment balance regularization

This module is active only when `proto_k > 0`.

---

## Reproducibility

The training script includes explicit seed control for:

- Python random seed
- NumPy seed
- PyTorch seed
- CUDA seed

It also uses deterministic CuDNN settings to improve reproducibility.

Recommended fixed settings for manuscript experiments should be recorded explicitly in the run command and output directory naming.

---

## Notes

- This module is designed for fixed train/validation split training, not 5-fold cross-validation.
- Internal validation is used for checkpoint selection and hyperparameter tuning.
- External validation should be performed separately after model development.
- Absolute paths in example commands should be replaced with environment-specific paths.
- Upstream data preparation and DINOv3 feature extraction are documented in other directories of the repository.
- Raw clinical data, transcriptomic data, and WSIs are not redistributed in this repository.

---

## Suggested citation context

This directory corresponds to the survival-modeling component of the PRISM framework and was used to generate:

- internal validation risk predictions
- best-epoch model checkpoints
- attention-based interpretability outputs
- downstream survival-analysis inputs for manuscript figures and tables
