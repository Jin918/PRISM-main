# SFigure8

Supplementary Figure 8 panel wrappers

## Builder

- kind: python
- script: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/make_sfigure8_style_refined.py

## Notes

Supplementary Figure 8 panel outputs should follow the finalized composite layout in Supplementary_Figure_S8_final.pdf. The wrappers therefore validate the panel PDFs exported under SFigure8_pdf_exports from that finalized build.

## Panels

- A: Cross-K anchor pathway allocation maps
  wrapper: panel_A.py
  builder: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/make_sfigure8_style_refined.py
  origin: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/Supplementary_Figure_S8_final.pdf
  outputs: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/SFigure8_pdf_exports/Supplementary_Figure_S8_panel_A.pdf
- B: Pathway allocation concentration across K
  wrapper: panel_B.py
  builder: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/make_sfigure8_style_refined.py
  origin: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/Supplementary_Figure_S8_final.pdf
  outputs: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/SFigure8_pdf_exports/Supplementary_Figure_S8_panel_B.pdf
- C: Fold-wise prototype burden shift summary
  wrapper: panel_C.py
  builder: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/make_sfigure8_style_refined.py
  origin: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/Supplementary_Figure_S8_final.pdf
  outputs: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/SFigure8_pdf_exports/Supplementary_Figure_S8_panel_C.pdf
- D: Held-out and external discrimination across K
  wrapper: panel_D.py
  builder: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/make_sfigure8_style_refined.py
  origin: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/Supplementary_Figure_S8_final.pdf
  outputs: /results/supplementary_figures/SFigure8_prototype_interpretation_robustness/SFigure8_pdf_exports/Supplementary_Figure_S8_panel_D.pdf
