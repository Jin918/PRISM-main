# SFigure6

Supplementary Figure 6 panel wrappers

## Builder

- kind: python
- script: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py

## Notes

Supplementary Figure 6 already exports panel-specific PDF/PNG/SVG files; the Final-tif layer simply adds stable panel entry points.

## Panels

- A: TCGA high-risk-associated pathway programs
  wrapper: panel_A.py
  builder: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py
  origin: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py
  outputs: /results/supplementary_figures/SFigure6_pathway_biology/Supplementary_Figure_6_pathway_biology_panel_A.pdf, /results/supplementary_figures/SFigure6_pathway_biology/Supplementary_Figure_6_pathway_biology_panel_A.png
- B: TCGA low-risk-associated pathway programs
  wrapper: panel_B.py
  builder: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py
  origin: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py
  outputs: /results/supplementary_figures/SFigure6_pathway_biology/Supplementary_Figure_6_pathway_biology_panel_B.pdf, /results/supplementary_figures/SFigure6_pathway_biology/Supplementary_Figure_6_pathway_biology_panel_B.png
- C: CPTAC external directional pathway-score support
  wrapper: panel_C.py
  builder: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py
  origin: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py
  outputs: /results/supplementary_figures/SFigure6_pathway_biology/Supplementary_Figure_6_pathway_biology_panel_C.pdf, /results/supplementary_figures/SFigure6_pathway_biology/Supplementary_Figure_6_pathway_biology_panel_C.png
- D: Cross-cohort concordance of pathway-score differences
  wrapper: panel_D.py
  builder: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py
  origin: /results/supplementary_figures/SFigure6_pathway_biology/make_supplementary_figure_6_pathway_biology.py
  outputs: /results/supplementary_figures/SFigure6_pathway_biology/Supplementary_Figure_6_pathway_biology_panel_D.pdf, /results/supplementary_figures/SFigure6_pathway_biology/Supplementary_Figure_6_pathway_biology_panel_D.png
