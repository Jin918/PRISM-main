# Figure3

Main-text Figure 3 panel wrappers

## Builder

- kind: rscript
- script: /Figure-main/script/make_figure3.R

## Notes

Panel A and B are emitted by the shared Figure 3 publication core. Panel C is the published 36-month AUC step-up bar chart.

## Panels

- A: TCGA clinical-integration ROC at 36 months
  wrapper: panel_A.py
  builder: /Figure-main/script/make_figure3.R
  origin: /Figure-main/script/figure3_publication_core.R
  outputs: /Figure-main/Figure3/Figure3A_TCGA_clinical_integration_ROC_36m.pdf, /Figure-main/Figure3/Figure3A_TCGA_clinical_integration_ROC_36m.png
- B: CPTAC clinical-integration ROC at 36 months
  wrapper: panel_B.py
  builder: /Figure-main/script/make_figure3.R
  origin: /Figure-main/script/figure3_publication_core.R
  outputs: /Figure-main/Figure3/Figure3B_CPTAC_clinical_integration_ROC_36m.pdf, /Figure-main/Figure3/Figure3B_CPTAC_clinical_integration_ROC_36m.png
- C: Landmark AUC step-up summary
  wrapper: panel_C.py
  builder: /Figure-main/script/make_figure3.R
  origin: /Figure-main/script/figure3C_auc_stepup_core.R
  outputs: /Figure-main/Figure3/Figure3C_AUC_stepup_36m.pdf, /Figure-main/Figure3/Figure3C_AUC_stepup_36m.png
