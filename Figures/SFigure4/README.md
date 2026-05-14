# SFigure4

Supplementary Figure 4 panel wrappers

## Builder

- kind: rscript
- script: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R

## Notes

Supplementary Figure 4 is organized as paired calibration/DCA panels: A/C/E are TCGA full-cohort, TCGA validation, and CPTAC calibration curves; B/D/F are the matched DCA curves.

## Panels

- A: TCGA full-cohort calibration
  wrapper: panel_A.py
  builder: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  origin: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  outputs: /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4_TCGA_all_cohort_calibration.pdf, /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4_TCGA_all_cohort_calibration.png
- B: TCGA full-cohort DCA
  wrapper: panel_B.py
  builder: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  origin: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  outputs: /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4_TCGA_all_cohort_DCA.pdf, /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4_TCGA_all_cohort_DCA.png
- C: TCGA validation 36-month calibration
  wrapper: panel_C.py
  builder: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  origin: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  outputs: /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4B_TCGA_validation_calibration.pdf, /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4B_TCGA_validation_calibration.png
- D: TCGA validation DCA at 36 months
  wrapper: panel_D.py
  builder: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  origin: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  outputs: /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4D_TCGA_validation_DCA.pdf, /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4D_TCGA_validation_DCA.png
- E: CPTAC external 36-month calibration
  wrapper: panel_E.py
  builder: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  origin: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  outputs: /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4C_CPTAC_external_calibration.pdf, /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4C_CPTAC_external_calibration.png
- F: CPTAC external DCA at 36 months
  wrapper: panel_F.py
  builder: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  origin: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  outputs: /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4E_CPTAC_external_DCA.pdf, /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4E_CPTAC_external_DCA.png
