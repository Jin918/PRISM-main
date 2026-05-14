# SFigure3

Supplementary Figure 3 panel wrappers

## Builder

- kind: python
- script: /Figure-main/script/make_supplementary_figures.py
- args: --figures S3

## Notes

Supplementary Figure 3 is rendered from real source tables inside the supplementary builder, and the builder now emits standalone panel exports A-B alongside the assembled figure.

## Panels

- A: C-index comparison across cohorts
  wrapper: panel_A.py
  builder: /Figure-main/script/make_supplementary_figures.py
  origin: /final_figure/Supplementary/scripts/build_supplementary_figures.py::render_s3
  outputs: /final_figure/Supplementary/revised_final/Supplementary_Figure_3_full_comparator_performance_panel_A.pdf, /final_figure/Supplementary/revised_final/Supplementary_Figure_3_full_comparator_performance_panel_A.png
- B: Landmark time-dependent AUC summary
  wrapper: panel_B.py
  builder: /Figure-main/script/make_supplementary_figures.py
  origin: /final_figure/Supplementary/scripts/build_supplementary_figures.py::render_s3
  outputs: /final_figure/Supplementary/revised_final/Supplementary_Figure_3_full_comparator_performance_panel_B.pdf, /final_figure/Supplementary/revised_final/Supplementary_Figure_3_full_comparator_performance_panel_B.png
