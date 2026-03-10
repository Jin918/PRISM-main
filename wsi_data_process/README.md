# WSI Data Processing

This directory contains the whole-slide image (WSI) preprocessing scripts used in the PRISM pipeline. These scripts generate slide-level overview images, tissue masks, and patch-level image tiles from TCGA-UCEC and CPTAC-UCEC H&E diagnostic WSIs.

The current implementation unifies TCGA and CPTAC preprocessing into three function-oriented scripts while preserving the original image-processing logic, input/output conventions, and downstream compatibility.

---

## Directory structure

```text
wsi_data_process/
├── 01_generate_overview.py
├── 02_generate_tissue_mask.py
├── 03_extract_patches.py
└── README.md


