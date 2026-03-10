# WSI Data Processing

This module contains the whole-slide image (WSI) preprocessing scripts used in the PRISM pipeline. It converts raw H&E diagnostic WSIs from TCGA-UCEC and CPTAC-UCEC into standardized patch-level image inputs for downstream self-supervised feature extraction and multimodal survival modeling.

The implementation preserves the original project logic, output format, and downstream compatibility while organizing preprocessing into three reusable steps.

## Files

~~~text
wsi_data_process/
├── 01_generate_overview.py
├── 02_generate_tissue_mask.py
├── 03_extract_patches.py
└── README.md
~~~

## Workflow

1. **Generate overview images**  
   Export low-resolution slide-level `.jpg` images for quick inspection and quality control.

2. **Generate tissue masks**  
   Detect tissue regions on low-resolution WSIs and save binary masks as `.npy` files, with `.png` visualizations.

3. **Extract patches**  
   Extract `256 × 256` image patches at 20×-equivalent resolution from tissue regions for downstream DINOv3 feature extraction and PRISM modeling.

## Scripts

### `01_generate_overview.py`
Generates slide-level overview `.jpg` images from WSIs at a specified OpenSlide level.

**Main features**
- Recursive WSI search
- Optional cohort filtering by clinical IDs
- Optional suffix filtering (for example, `-21.svs` for CPTAC)
- Skip existing outputs
- Robust handling of unreadable WSI files

**Output**
- One `.jpg` overview image per WSI

### `02_generate_tissue_mask.py`
Generates tissue masks from low-resolution WSIs and saves:
- binary masks as `.npy`
- mask visualizations as `.png`
- thumbnails as `.png`

**Masking logic**
- RGB-channel Otsu thresholding
- Background exclusion
- Dark-region filtering with `RGB_min`
- Final mask stored as a boolean array

**Main features**
- Supports both TCGA and CPTAC
- Optional filtering by text ID list or clinical CSV
- Multi-process execution
- Skip existing masks
- Optional keep/missing case logs

**Outputs**
- `mask_npy/*.npy`
- `mask_png/*.png`
- `thumb/*.png`

### `03_extract_patches.py`
Extracts patch images from WSI tissue regions using precomputed tissue masks.

**Patch extraction logic**
- Target magnification: 20× equivalent
- Patch size: `256 × 256`
- Extraction restricted to mask-positive coordinates
- WSI-specific coordinate conversion using `MPP_X`
- Per-slide output folders
- `.done` markers for resumable execution

**Main features**
- Supports both TCGA and CPTAC
- Optional filtering by text ID list or clinical CSV
- Multi-process extraction
- Automatic skip/recovery for completed slides

**Outputs**
- `patch_root/<WSI_NAME>/*.png`
- `patch_root/<WSI_NAME>/.done`

## Requirements

### Software
- Python 3
- `openslide-python`
- OpenSlide system library
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-image`

### Input data
- H&E diagnostic WSIs in `.svs` format
- Optional clinical ID list or clinical CSV for filtering
- Precomputed tissue masks for patch extraction

### Supported case IDs
- `TCGA-XX-XXXX`
- `C3L-00001`
- `C3N-00001`

## Example Usage

### 1. Generate overview images

~~~bash
python 01_generate_overview.py \
  --wsi_path /root/code/PRISM-main/datasets/svs/TCGA-UCEC \
  --out_dir /Pathology_data_2/UCEC_new/TCGA/level1_jpg \
  --level 1 \
  --suffix .svs
~~~

~~~bash
python 01_generate_overview.py \
  --wsi_path /root/code/PRISM-main/datasets/svs/CPTAC-UCEC \
  --out_dir /Pathology_data_2/UCEC_new/CPTAC/level1_jpg \
  --level 1 \
  --suffix -21.svs \
  --clinical_ids /Pathology_data_2/UCEC_external/clinical_ids.txt
~~~

### 2. Generate tissue masks

~~~bash
python 02_generate_tissue_mask.py \
  --wsi_path /root/code/PRISM-main/datasets/svs/TCGA-UCEC \
  --npy_path /root/code/PRISM-main/UCEC_DINOV3/mask_npy \
  --gray_path /root/code/PRISM-main/UCEC_DINOV3/mask_png \
  --thumb_path /root/code/PRISM-main/UCEC_DINOV3/thumb \
  --level 2 \
  --RGB_min 50 \
  --workers 8 \
  --suffix .svs
~~~

~~~bash
python 02_generate_tissue_mask.py \
  --wsi_path /root/code/PRISM-main/datasets/svs/CPTAC-UCEC \
  --out_base /Pathology_data_2/UCEC_new/CPTAC \
  --npy_path /Pathology_data_2/UCEC_new/CPTAC/mask_npy \
  --gray_path /Pathology_data_2/UCEC_new/CPTAC/mask_png \
  --thumb_path /Pathology_data_2/UCEC_new/CPTAC/thumb \
  --level 2 \
  --RGB_min 50 \
  --workers 8 \
  --suffix -21.svs \
  --clin_csv /root/code/PRISM-main/datasets/prism_481/datasetsnew/test_88/CPTAC_UCEC_clinic_88.csv \
  --id_col ID
~~~

### 3. Extract patches

~~~bash
python 03_extract_patches.py \
  --wsi_root /root/code/PRISM-main/datasets/svs/TCGA-UCEC \
  --mask_root /Pathology_data_2/UCEC_new/TCGA/mask_npy \
  --patch_root /Pathology_data_2/UCEC_new/TCGA/patch_data_whole_slide_20X_256 \
  --patch_size 256 \
  --target_level 20 \
  --num_process 80 \
  --suffix .svs
~~~

~~~bash
python 03_extract_patches.py \
  --wsi_root /root/code/PRISM-main/datasets/svs/CPTAC-UCEC \
  --mask_root /Pathology_data_2/UCEC_new/CPTAC/mask_npy \
  --patch_root /Pathology_data_2/UCEC_new/CPTAC/patch_data_whole_slide_20X_256 \
  --patch_size 256 \
  --target_level 20 \
  --num_process 80 \
  --suffix -21.svs
~~~

## Key Parameters

### Common filtering
- `--suffix`: restrict WSI files by suffix
- `--clinical_ids` / `--clinical_ids_txt`: text file with one case ID per line
- `--clin_csv` and `--id_col`: clinical CSV-based filtering

### Tissue masking
- `--level`: OpenSlide level for tissue detection
- `--RGB_min`: minimum RGB threshold for dark-region removal
- `--workers`: number of worker processes

### Patch extraction
- `--patch_size`: patch size in pixels, default `256`
- `--target_level`: target magnification equivalent, default `20`
- `--num_process`: number of worker processes

## Downstream Usage

The outputs generated here are used in the downstream PRISM workflow for:
- slide-level quality control
- tissue-restricted patch extraction
- DINOv3 feature extraction
- patch-level embedding clustering
- fixed-length WSI token construction
- multimodal survival modeling

## Reproducibility Notes

- Existing outputs are skipped by default.
- Patch extraction uses per-slide `.done` markers for resumable execution.
- Filtering by cohort IDs can be enabled or disabled depending on the analysis setting.
- For CPTAC, restricting to `-21.svs` slides is commonly used in this project.

## Recommended Output Layout

~~~text
/Pathology_data_2/UCEC_new/
├── TCGA/
│   ├── level1_jpg/
│   ├── mask_npy/
│   ├── mask_png/
│   ├── thumb/
│   └── patch_data_whole_slide_20X_256/
└── CPTAC/
    ├── level1_jpg/
    ├── mask_npy/
    ├── mask_png/
    ├── thumb/
    └── patch_data_whole_slide_20X_256/
~~~

## Summary

This module standardizes pathology image preprocessing in PRISM into three reusable steps:

- `01_generate_overview.py`
- `02_generate_tissue_mask.py`
- `03_extract_patches.py`

Together, these scripts convert raw WSIs into patch-level image inputs suitable for downstream self-supervised feature extraction and interpretable multimodal survival analysis.
