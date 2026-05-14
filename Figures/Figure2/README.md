# Figure2

Main-text Figure 2 panel wrappers

## Builder

- kind: rscript
- script: /Figure-main/script/make_figure2.R

## Notes

Each wrapper targets one panel but delegates to the existing Figure 2 manuscript wrapper, which rebuilds the full figure and then validates the requested panel outputs.

## Panels

- A: C-index comparison across cohorts
  wrapper: panel_A.py
  builder: /Figure-main/script/make_figure2.R
  origin: /Figure-main/script/figure2A_cindex_barplot.R
  outputs: /Figure-main/Figure2/Figure2A_cindex_comparison.pdf, /Figure-main/Figure2/Figure2A_cindex_comparison.png
- B: TCGA training ROC at 36 months
  wrapper: panel_B.py
  builder: /Figure-main/script/make_figure2.R
  origin: /Figure-main/script/figure2B_train_roc.R
  outputs: /Figure-main/Figure2/Figure2B_tdROC_TCGA_training_36m.pdf, /Figure-main/Figure2/Figure2B_tdROC_TCGA_training_36m.png
- C: TCGA internal-validation ROC at 36 months
  wrapper: panel_C.py
  builder: /Figure-main/script/make_figure2.R
  origin: /Figure-main/script/figure2C_vali_roc.R
  outputs: /Figure-main/Figure2/Figure2C_tdROC_TCGA_internal_validation_36m.pdf, /Figure-main/Figure2/Figure2C_tdROC_TCGA_internal_validation_36m.png
- D: CPTAC external-validation ROC at 36 months
  wrapper: panel_D.py
  builder: /Figure-main/script/make_figure2.R
  origin: /Figure-main/script/figure2D_cptac_roc.R
  outputs: /Figure-main/Figure2/Figure2D_tdROC_CPTAC_external_validation_36m.pdf, /Figure-main/Figure2/Figure2D_tdROC_CPTAC_external_validation_36m.png
