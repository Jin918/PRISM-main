# Datasets

This directory contains the scripts and processed files used to generate the core inputs for the main PRISM analyses.

## Directory structure

- `data_processing/`  
  Scripts for preprocessing TCGA-UCEC, CPTAC-UCEC, and MSigDB data.

- `model_input_building/`  
  Scripts for converting processed cohort tables and pathway matrices into model-ready inputs for PRISM training.

- `processed/`  
  Core processed files required for downstream analyses.

## Data processing scripts

### `data_processing/00_MSigDB_data_processing.R`
Builds the MSigDB gene membership matrix from the selected gene sets and exports:

- `MSigDB_2sets_co_genes.csv`

### `data_processing/01_TCGA_data_processing.R`
Processes TCGA-UCEC clinical and transcriptomic data, applies study inclusion/exclusion criteria, performs pathway-level normalization, and exports:

- `TCGA_UCEC_clin_PFS_287.csv`
- `TCGA_UCEC_log_FPKM_pathway_normolized.csv`

### `data_processing/02_CPTAC_data_processing.R`
Processes CPTAC-UCEC clinical and transcriptomic data, applies study inclusion/exclusion criteria, performs pathway-level normalization, and exports:

- `CPTAC_UCEC_clinic_63.csv`
- `CPTAC_UCEC_log_normalized.csv`

## Model input building scripts

### `model_input_building/04_make_cox_all_txt.py`
Converts the processed TCGA clinical table into Cox-format survival text files for downstream model development.

### `model_input_building/05_make_split.py`
Generates the fixed train/validation split used in internal model development.

### `model_input_building/06_make_pathway_gene_matrix.py`
Builds patient-level pathway-by-gene matrices for the PRISM training pipeline.

## Core processed files

The following files are required inputs for the downstream PRISM analyses:

| File | Description |
|---|---|
| `TCGA_UCEC_clin_PFS_287.csv` | TCGA-UCEC clinical table for the model-development cohort |
| `CPTAC_UCEC_clinic_63.csv` | CPTAC-UCEC clinical table for the external validation cohort |
| `TCGA_UCEC_log_FPKM_pathway_normolized.csv` | TCGA-UCEC pathway-level normalized expression matrix |
| `CPTAC_UCEC_log_normalized.csv` | CPTAC-UCEC pathway-level normalized expression matrix |
| `MSigDB_2sets_co_genes.csv` | MSigDB gene membership matrix used for pathway projection |

## Notes

- These files constitute the core processed inputs used in the main manuscript analyses.
- Raw TCGA and CPTAC data should be obtained from their original sources and are not redistributed in this repository.
- The scripts in this directory document the preprocessing workflow from raw data to analysis-ready inputs and then to model-ready files.
