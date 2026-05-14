# Figure5

Main-text Figure 5 panel wrappers

## Builder

- kind: python
- script: /Figure-main/script/make_figure5.py

## Notes

The Figure 5 wrapper refreshes the revised_final source directory and syncs the manuscript-facing panel PDFs into Figure-main/Figure5. Panel descriptions follow the checked-in Figure5 source manifest.

## Panels

- A: Cross-cohort pathway-prototype alignment landscape
  wrapper: panel_A.py
  builder: /Figure-main/script/make_figure5.py
  origin: /Figure-main/Figure5/revised_final/Figure5_source_manifest.tsv
  outputs: /Figure-main/Figure5/Figure5A_cross_cohort_alignment.pdf
- B: TCGA-versus-CPTAC concordance scatter
  wrapper: panel_B.py
  builder: /Figure-main/script/make_figure5.py
  origin: /Figure-main/Figure5/revised_final/Figure5_source_manifest.tsv
  outputs: /Figure-main/Figure5/Figure5B_alignment_concordance.pdf
- C: Representative morphology of learned prototypes in TCGA-UCEC
  wrapper: panel_C.py
  builder: /Figure-main/script/make_figure5.py
  origin: /Figure-main/Figure5/revised_final/Figure5_source_manifest.tsv
  outputs: /Figure-main/Figure5/Figure5C_TCGA_prototype_morphology_manual_selected.pdf
- D: External morphological validation of learned prototypes in CPTAC-UCEC
  wrapper: panel_D.py
  builder: /Figure-main/script/make_figure5.py
  origin: /Figure-main/Figure5/revised_final/Figure5_source_manifest.tsv
  outputs: /Figure-main/Figure5/Figure5D_CPTAC_prototype_morphology_manual_selected_publication_ready.pdf
- E: One pathway shown across TCGA/CPTAC low/high-risk case blocks
  wrapper: panel_E.py
  builder: /Figure-main/script/make_figure5.py
  origin: /Figure-main/Figure5/revised_final/Figure5_source_manifest.tsv
  outputs: /Figure-main/Figure5/Figure5E_within_case_prototype_localization_largefont.pdf
- F: Prototype-centered morphology panel using filtered TCGA/CPTAC patch exemplars
  wrapper: panel_F.py
  builder: /Figure-main/script/make_figure5.py
  origin: /Figure-main/Figure5/revised_final/Figure5_source_manifest.tsv
  outputs: /Figure-main/Figure5/Figure5F_representative_patch_burden_boxplot.pdf
