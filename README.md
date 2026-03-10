# PRISM-main 🌈⃤

# Data preprocessing

This directory contains the preprocessing scripts used to generate the core input files required for the main PRISM analyses. These scripts prepare the TCGA-UCEC and CPTAC-UCEC clinical cohort tables, pathway-level transcriptomic matrices, and the MSigDB gene membership matrix used for pathway projection.

## Scripts

### `00_MSigDB_data_processing.R`
Builds the MSigDB gene membership matrix from the selected gene sets and exports:

- `MSigDB_2sets_co_genes.csv`

### `01_TCGA_data_processing.R`
Processes TCGA-UCEC clinical and transcriptomic data, applies study inclusion/exclusion criteria, performs pathway-level normalization, and exports:

- `EC_Clinic_PFS_287.csv`
- `TCGA_UCEC_log_FPKM_pathway_normolized.csv`

### `02_CPTAC_data_processing.R`
Processes CPTAC-UCEC clinical and transcriptomic data, applies study inclusion/exclusion criteria, performs pathway-level normalization, and exports:

- `CPTAC_UCEC_clinic_63.csv`
- `CPTAC_UCEC_log_normalized.csv`

## Core output files

The following files are required inputs for the downstream PRISM analyses:

| File | Description |
|---|---|
| `EC_Clinic_PFS_287.csv` | TCGA-UCEC clinical table for the model-development cohort |
| `CPTAC_UCEC_clinic_63.csv` | CPTAC-UCEC clinical table for the external validation cohort |
| `TCGA_UCEC_log_FPKM_pathway_normolized.csv` | TCGA-UCEC pathway-level normalized expression matrix |
| `CPTAC_UCEC_log_normalized.csv` | CPTAC-UCEC pathway-level normalized expression matrix |
| `MSigDB_2sets_co_genes.csv` | MSigDB gene membership matrix used for pathway projection |

## Notes

- These files constitute the core processed inputs used in the main manuscript analyses.
- Raw TCGA and CPTAC data should be obtained from their original sources and are not redistributed in this repository.
- The scripts in this directory are intended to document the preprocessing workflow from raw data to analysis-ready inputs.
