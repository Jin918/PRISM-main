# Figure-main/Final-tif

This folder now serves two related purposes:

- manuscript TIFF deliverables exported from the manuscript-facing Figure-main outputs
- a panel-organized code layer under per-figure folders such as `Figure2/`, `Figure5/`, and `SFigure7/`

## Scope

- Stable TIFF wrapper: [Figure-main/script/make_final_tif.py](/root/code/PRISM-main/Figure-main/script/make_final_tif.py)
- Stable panel-layout wrapper: [Figure-main/script/make_final_tif_panels.py](/root/code/PRISM-main/Figure-main/script/make_final_tif_panels.py)
- Stable Result PDF wrapper: [Figure-main/script/make_final_tif_result_pdfs.py](/root/code/PRISM-main/Figure-main/script/make_final_tif_result_pdfs.py)
- Current managed set: Figure2-Figure5 and SFigure3-SFigure8
- Current legacy holdouts: Figure1.tif and SFigure1.tif remain curated outputs and are not yet wired to the new wrappers

## Usage

From the repository root:

```bash
python Figure-main/script/make_final_tif.py
python Figure-main/script/make_final_tif.py --figures Figure5,S5,S7

python Figure-main/script/make_final_tif_panels.py
python Figure-main/script/make_final_tif_panels.py --figures Figure2,Figure5,S2,S4,S8

python Figure-main/script/make_final_tif_result_pdfs.py
python Figure-main/script/make_final_tif_result_pdfs.py --figures Figure2,Figure5,S2,S8
```

`make_final_tif.py` refreshes manuscript-facing composite PNG sources and exports the checked-in TIFF delivery files at the current delivery size and 300 DPI into this folder.

`make_final_tif_panels.py` materializes a folder-per-figure code layer with:

- `panel_<letter>.py` thin wrappers for each manuscript panel
- `README.md` per figure describing the panel mapping
- `panel_manifest.tsv` per figure plus a top-level `panel_index.tsv`

`make_final_tif_result_pdfs.py` refreshes the selected reproducible figure outputs and copies each panel PDF into [Figure-main/Final-tif/Result](/root/code/PRISM-main/Figure-main/Final-tif/Result) under per-figure subfolders.

## Panel-Layer Notes

- Main-text panel wrappers delegate to the existing Figure-main entry scripts and validate the corresponding panel outputs.
- Supplementary figures S4-S8 use the checked-in panel-specific generators and checked-in panel outputs where available.
- S3 now validates its standalone panel exports from the supplementary builder.
- SFigure4 is organized as six calibration/DCA panels: A/C/E are full-TCGA, validation-TCGA, and CPTAC calibration curves; B/D/F are the matched DCA curves.
- SFigure5 is organized as a three-panel set: panel A is the finalized nomogram, and panels B/C are the preserved feature-attribution bar plots.
- SFigure8 panel exports should follow the finalized composite layout in Supplementary_Figure_S8_final.pdf; the Final-tif layer therefore targets the panel PDFs under results/supplementary_figures/SFigure8_prototype_interpretation_robustness/SFigure8_pdf_exports.

## Notes

- Main-text TIFF export depends on [Figure-main/Figure2](/root/code/PRISM-main/Figure-main/Figure2), [Figure-main/Figure3](/root/code/PRISM-main/Figure-main/Figure3), [Figure-main/Figure4](/root/code/PRISM-main/Figure-main/Figure4), and [Figure-main/Figure5](/root/code/PRISM-main/Figure-main/Figure5).
- Supplementary TIFF export depends on [Figure-main/Supplementary](/root/code/PRISM-main/Figure-main/Supplementary).
- The TIFF wrapper continues to reproduce the existing delivery geometry from the curated manuscript outputs rather than introducing a new publication specification.