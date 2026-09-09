# PRISM 🌈⃤

Prototype-resolved multimodal survival modelling for early-stage endometrioid endometrial carcinoma.

[Model weights](https://github.com/Jin918/PRISM-main/releases/tag/v1.0-prism) | [MIT License](LICENSE)

## Overview

PRISM is a multimodal survival-modelling framework that integrates histopathological whole-slide image features with pathway-level transcriptomic representations. It was developed to estimate progression-free survival (PFS) risk in FIGO stage I–II endometrioid endometrial carcinoma while retaining prototype-level morphological and pathway-level interpretability.

The model was developed using TCGA-UCEC and independently evaluated using CPTAC-UCEC. Because the external cohort contained a limited number of PFS events, the CPTAC-UCEC analysis should be interpreted as an exploratory independent external evaluation rather than definitive clinical validation.

This repository accompanies the PRISM manuscript and provides the core code for:

* clinical and transcriptomic preprocessing;
* WSI tissue detection and patch extraction;
* pathology-domain DINOv3 adaptation and feature extraction;
* cluster-guided patch selection;
* multimodal and unimodal survival-model training;
* inference and prototype-assignment export;
* statistical evaluation and manuscript figure generation; and
* access to the released PRISM model weights.

PRISM is intended for research use only and is not a clinically approved prognostic system.

## Study cohorts

| Cohort     |                                        Role | Patients | PFS events |
| ---------- | ------------------------------------------: | -------: | ---------: |
| TCGA-UCEC  |   Model development and internal validation |      287 |         53 |
| CPTAC-UCEC | Exploratory independent external evaluation |       63 |          8 |

No CPTAC-UCEC sample was used to fit the final released PRISM model.

## Repository structure

| Directory                                                    | Description                                                                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| [`datasets/`](datasets/)                                     | Clinical and transcriptomic preprocessing, cohort construction, and generation of model-ready pathway inputs |
| [`wsi_data_process/`](wsi_data_process/)                     | WSI overview generation, tissue masking, and patch extraction                                                |
| [`dinov3/`](dinov3/)                                         | Pathology-domain DINOv3 adaptation, patch-feature extraction, and proxy evaluation                           |
| [`patch_clustering/`](patch_clustering/)                     | Within-slide feature clustering and representative patch selection                                           |
| [`apl_pathway_wsi_predict/`](apl_pathway_wsi_predict/)       | PRISM training, ablation models, inference, and prototype-assignment export                                  |
| [`clinical_evaluation/`](clinical_evaluation/)               | Clinical comparators, discrimination, calibration, decision-curve, and survival analyses                     |
| [`prototype_interpretability/`](prototype_interpretability/) | Prototype localisation and morphomolecular interpretation analyses                                           |
| [`technique_evaluation/`](technique_evaluation/)             | Robustness, sensitivity, and technical evaluation scripts                                                    |
| [`Figures/`](Figures/)                                       | Scripts and metadata used to generate the manuscript figures                                                 |

Detailed input formats and example commands are provided in the README file within each module.

## Analysis workflow

The main workflow is:

1. Obtain TCGA-UCEC and CPTAC-UCEC clinical, transcriptomic, and diagnostic WSI data from their original repositories.
2. Construct the eligible study cohorts and harmonise clinical variables.
3. Process transcriptomic data and align genes to the selected MSigDB pathway programmes.
4. Generate tissue masks and extract WSI patches.
5. Adapt the DINOv3 encoder to pathology images and extract patch-level representations.
6. Cluster patch representations within each WSI and select a fixed set of representative patches.
7. Construct patient-level pathway-by-gene matrices.
8. Train PRISM and the prespecified ablation models using the TCGA-UCEC development data.
9. Refit the locked PRISM configuration using the complete TCGA-UCEC cohort.
10. Apply the refitted model to CPTAC-UCEC without model refitting or outcome-based threshold optimisation.
11. Perform discrimination, calibration, decision-curve, survival, and prototype-resolved interpretation analyses.

## Data access

Raw TCGA and CPTAC clinical data, RNA-sequencing data, and WSIs are not redistributed in this repository. They should be obtained from the Genomic Data Commons and CPTAC/PDC under their respective access and usage conditions.

MSigDB gene sets must be obtained separately in accordance with the MSigDB licence. The scripts in [`datasets/data_processing/`](datasets/data_processing/) convert these source files into the matrices required by PRISM.

### Expected processed inputs

The preprocessing workflow generates the following principal files:

| File                                        | Description                                                 |
| ------------------------------------------- | ----------------------------------------------------------- |
| `EC_Clinic_PFS_287.csv`                     | TCGA-UCEC clinical and PFS table for the development cohort |
| `CPTAC_UCEC_clinic_63.csv`                  | CPTAC-UCEC clinical and PFS table for external evaluation   |
| `TCGA_UCEC_log_FPKM_pathway_normolized.csv` | TCGA-UCEC pathway-gene expression matrix                    |
| `CPTAC_UCEC_log_normalized.csv`             | CPTAC-UCEC pathway-gene expression matrix                   |
| `MSigDB_2sets_co_genes.csv`                 | Binary pathway-gene membership matrix                       |

These filenames are used consistently by the downstream model-input scripts.

Patient identifiers and input filenames must be harmonised before model training. TCGA barcodes are reduced to the patient-level identifier where required; CPTAC identifiers retain the corresponding `C3L-` or `C3N-` patient identifier.

## Software environment

The manuscript analyses were conducted using:

* Python 3.10;
* R 4.3.3;
* PyTorch;
* OpenSlide;
* NumPy and pandas;
* scikit-learn and lifelines;
* matplotlib and related plotting packages; and
* the R survival-analysis packages listed in the manuscript’s Supplementary Table S12.

To create the tested environment:

```bash
git clone https://github.com/Jin918/PRISM-main.git
cd PRISM-main

conda env create -f environment.yml
conda activate prism
```

R dependencies can be restored using:

```bash
R -e "renv::restore()"
```

The official DINOv3 implementation and its pretrained weights must be obtained separately. The exact upstream version and checkpoint used by PRISM should remain fixed when reproducing the pathology feature-extraction workflow.

## Data preprocessing

Run the preprocessing scripts in numerical order:

```bash
Rscript datasets/data_processing/00_MSigDB_data_processing.R
Rscript datasets/data_processing/01_TCGA_data_processing.R
Rscript datasets/data_processing/02_CPTAC_data_processing.R

python datasets/model_input_building/04_make_cox_all_txt.py
python datasets/model_input_building/05_make_split.py --seed 302
python datasets/model_input_building/06_make_pathway_gene_matrix.py
```

The primary split and the repeated-split robustness analyses use the same event-stratified splitting procedure. Alternative seeds can be supplied through `05_make_split.py`.

See [`datasets/README.md`](datasets/README.md) for required source files, cohort filters, output definitions, and directory organisation.

## WSI processing

The WSI workflow consists of:

```text
Diagnostic WSI
  → low-resolution overview
  → tissue mask
  → 256 × 256 patches at 20×-equivalent resolution
  → DINOv3 patch representations
  → within-slide clustering
  → 500 representative WSI tokens
```

Detailed commands are provided in:

* [`wsi_data_process/README.md`](wsi_data_process/README.md)
* [`dinov3/README.md`](dinov3/README.md)
* [`patch_clustering/README.md`](patch_clustering/README.md)

The final pathology input for each patient contains 500 representative patch embeddings, each with 384 features.

## PRISM model training

The main model code is located in [`apl_pathway_wsi_predict/`](apl_pathway_wsi_predict/).

The available model configurations include:

* PRISM with multimodal integration and the adaptive pathology learning prototype bottleneck;
* an APL-ablated multimodal model;
* a WSI-only model;
* a pathway-only model; and
* additional contrastive-learning ablations.

The principal training entry point is:

```bash
python apl_pathway_wsi_predict/train_survival.py [arguments]
```

Inference on a target cohort is performed using:

```bash
python apl_pathway_wsi_predict/predict_survival.py [arguments]
```

Complete examples and argument definitions are provided in [`apl_pathway_wsi_predict/README.md`](apl_pathway_wsi_predict/README.md).

The locked PRISM configuration uses:

* 500 × 384 pathology-feature input;
* 236 × 6,292 pathway-gene input; and
* four adaptive pathology prototypes.

## Released model weights

The model files are available from [release `v1.0-prism`](https://github.com/Jin918/PRISM-main/releases/tag/v1.0-prism).

| File                             | Purpose                                                          |
| -------------------------------- | ---------------------------------------------------------------- |
| `PRISM_TCGAfull_n287_seed42.pth` | Final PRISM model state dictionary for inference                 |
| `PRISM_TCGAfull_n287_seed42.pt`  | Full training checkpoint, including model and optimisation state |
| `checkpoint0010.pth`             | Upstream pathology-domain feature-extraction checkpoint          |

The released PRISM model was refitted on the complete TCGA-UCEC cohort after the architecture and hyperparameters had been fixed. It is distinct from the split-specific model used to obtain the TCGA internal-validation results. Consequently, the released checkpoint should be used for independent application and external evaluation, not to reconstruct the split-specific internal-validation predictions.

## Statistical evaluation

The statistical analysis distinguishes among:

* split-specific TCGA training and internal-validation performance;
* apparent performance after refitting in the complete TCGA cohort;
* exploratory independent external performance in CPTAC-UCEC; and
* fixed-horizon incremental clinical-value analyses.

The principal evaluation measures include:

* Harrell’s C-index;
* IPCW-adjusted time-dependent AUC;
* paired bootstrap differences between models;
* calibration slope and calibration-in-the-large;
* decision-curve analysis;
* Kaplan–Meier analysis;
* Cox regression with restricted cubic splines; and
* prototype-resolved cross-cohort concordance analyses.

Risk groups are used for visualisation and log-rank comparisons. The continuous PRISM risk score is used for C-index, time-dependent AUC, and Cox regression analyses.

## Reproducibility notes

* The TCGA internal-validation results were obtained using split-specific models.
* The public PRISM checkpoint was refitted using all 287 eligible TCGA-UCEC patients.
* CPTAC-UCEC was not used for model fitting, hyperparameter selection, or external cut-point optimisation.
* Random seeds, cohort partitions, model parameters, and software versions are reported in the code and Supplementary Methods.
* Raw data and licensed third-party resources are not included.
* External estimates should be interpreted cautiously because CPTAC-UCEC contained only eight PFS events.

## Citation

The manuscript describing PRISM is currently under review. Full citation information will be added after publication.

## Licence

The source code in this repository is distributed under the [MIT License](LICENSE). Third-party data, software, gene-set resources, and pretrained model weights remain subject to their original licences and terms of use.

## Contact

Questions about the code or reproducibility can be submitted through the repository’s GitHub Issues page.

