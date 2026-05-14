# Figure4

Main-text Figure 4 panel wrappers

## Builder

- kind: rscript
- script: /Figure-main/script/make_figure4.R

## Notes

The existing Figure 4 core script emits all four main panels in one pass; each wrapper validates only its own panel outputs.

## Panels

- A: TCGA Kaplan-Meier stratification
  wrapper: panel_A.py
  builder: /Figure-main/script/make_figure4.R
  origin: /Figure-main/script/figure4_final_spline_forest_core.R
  outputs: /Figure-main/Figure4/Figure4A_TCGA_KM.pdf, /Figure-main/Figure4/Figure4A_TCGA_KM.png
- B: CPTAC Kaplan-Meier stratification
  wrapper: panel_B.py
  builder: /Figure-main/script/make_figure4.R
  origin: /Figure-main/script/figure4_final_spline_forest_core.R
  outputs: /Figure-main/Figure4/Figure4B_CPTAC_KM.pdf, /Figure-main/Figure4/Figure4B_CPTAC_KM.png
- C: TCGA spline and forest summary
  wrapper: panel_C.py
  builder: /Figure-main/script/make_figure4.R
  origin: /Figure-main/script/figure4_final_spline_forest_core.R
  outputs: /Figure-main/Figure4/Figure4C_TCGA_spline_forest.pdf, /Figure-main/Figure4/Figure4C_TCGA_spline_forest.png
- D: CPTAC spline and forest summary
  wrapper: panel_D.py
  builder: /Figure-main/script/make_figure4.R
  origin: /Figure-main/script/figure4_final_spline_forest_core.R
  outputs: /Figure-main/Figure4/Figure4D_CPTAC_spline_forest.pdf, /Figure-main/Figure4/Figure4D_CPTAC_spline_forest.png
