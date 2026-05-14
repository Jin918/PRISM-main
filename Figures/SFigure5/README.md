# SFigure5

Supplementary Figure 5 panel wrappers

## Builder

- kind: python
- script: /Figure-main/script/make_supplementary_figures.py
- args: --figures S4,S5

## Notes

Supplementary Figure 5 is now treated as a three-panel set: A uses the finalized integrated-model nomogram, while B and C preserve the previously finalized feature-attribution bar plots.

## Panels

- A: Final integrated prognostic nomogram
  wrapper: panel_A.py
  builder: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  origin: /results/supplementary_figures/SFigure4_clinical_implementation/make_supplementary_figure_4_clinical_implementation.R
  outputs: /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4A_nomogram.pdf, /results/supplementary_figures/SFigure4_clinical_implementation/SFigure4A_nomogram.png
- B: TCGA internal validation feature attribution
  wrapper: panel_B.py
  builder: /results/supplementary_figures/SFigure5_feature_attribution/make_supplementary_figure_5_feature_attribution.R
  origin: /results/supplementary_figures/SFigure5_feature_attribution/make_supplementary_figure_5_feature_attribution.R
  outputs: /results/supplementary_figures/SFigure5_feature_attribution/SFigure5A_TCGA_internal_feature_attribution.pdf, /results/supplementary_figures/SFigure5_feature_attribution/SFigure5A_TCGA_internal_feature_attribution.png
- C: CPTAC external feature attribution
  wrapper: panel_C.py
  builder: /results/supplementary_figures/SFigure5_feature_attribution/make_supplementary_figure_5_feature_attribution.R
  origin: /results/supplementary_figures/SFigure5_feature_attribution/make_supplementary_figure_5_feature_attribution.R
  outputs: /results/supplementary_figures/SFigure5_feature_attribution/SFigure5B_CPTAC_external_feature_attribution.pdf, /results/supplementary_figures/SFigure5_feature_attribution/SFigure5B_CPTAC_external_feature_attribution.png
