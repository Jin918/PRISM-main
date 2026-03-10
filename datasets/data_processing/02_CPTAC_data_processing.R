rm(list = ls())

library(pacman)
p_load(data.table,dplyr,stringr,tibble,readxl,readr,gtsummary,gt)

options(stringsAsFactors = FALSE)
options(scipen = 100)

trans_path   <- "/Users/jin/Desktop/UCEC-CPTAC-PRISM/DATA/clinic/CPTAC_63/raw/CPTAC_UCEC_transcriptomics_bcm.tsv"
clinc_path   <- "/Users/jin/Desktop/UCEC-CPTAC-PRISM/DATA/clinic/CPTAC_63/raw/CPTAC_UCEC_clinical_mssm.tsv"
clinc_path_2 <- "/Users/jin/Desktop/UCEC-CPTAC-PRISM/DATA/clinic/CPTAC_63/raw/NIHMS1566438-supplement-1.xlsx"
path_wsi_id  <- "/Users/jin/Desktop/UCEC-CPTAC-PRISM/DATA/clinic/CPTAC_63/raw/CPTAC_pathology_ID.csv"


## 01 Supplementary clinical table reading####
if (T) {
  clin_df2 <- read_excel(clinc_path_2, sheet = 1)
  clin_df2 <- as.data.frame(clin_df2)
  
  clin_df2_clean <- clin_df2 %>%
    mutate(
      Patient_ID = str_extract(Proteomics_Participant_ID, "C3[LN]-\\d{5}")
    ) %>%
    filter(!is.na(Patient_ID)) %>%
    filter(Proteomics_Tumor_Normal == "Tumor") %>%
    filter(idx != "S104") %>%
    select(idx, Patient_ID, Myometrial_invasion_Specify, LVSI, Genomics_subtype) %>%
    distinct()
  
  dim(clin_df2_clean)
  any(duplicated(clin_df2_clean$Patient_ID))
}


## 02 Main clinical table cleaning####
if (T) {
  clin_df <- fread(clinc_path, data.table = FALSE, check.names = FALSE)
  
  clin_df_filtered <- clin_df %>%
    dplyr::select(
      Patient_ID,
      age,
      histologic_type,
      histologic_grade,
      pathologic_staging_primary_tumor_pt,
      tumor_necrosis,
      residual_tumor,
      tumor_stage_pathological,
      history_of_cancer,
      `adjuvant_post-operative_radiation_therapy`,
      new_tumor_after_initial_treatment,
      number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_new_tumor_event_after_initial_treatment,
      number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_last_contact,
      number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_death,
      `Recurrence-free survival, days`,
      `Recurrence status (1, yes; 0, no)`,
      `Overall survival, days`,
      `Survival status (1, dead; 0, alive)`
    ) %>%
    dplyr::rename(
      Age = age,
      Histological_Type = histologic_type,
      Histologic_Grade = histologic_grade,
      Radiation_Therapy = `adjuvant_post-operative_radiation_therapy`,
      Figo_Stage = pathologic_staging_primary_tumor_pt,
      Tumor_Necrosis = tumor_necrosis,
      Residual_Tumor = residual_tumor,
      Tumor_Stage_Pathological = tumor_stage_pathological,
      Prior_Treatment = history_of_cancer,
      NTE_flag = new_tumor_after_initial_treatment,
      NTE.days = number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_new_tumor_event_after_initial_treatment,
      Last.days = number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_last_contact,
      Death.days = number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_death,
      RFS.time = `Recurrence-free survival, days`,
      RFS = `Recurrence status (1, yes; 0, no)`,
      OS.time = `Overall survival, days`,
      OS = `Survival status (1, dead; 0, alive)`
    ) %>%
    mutate(
      Histologic_Grade = str_extract(as.character(Histologic_Grade), "^[^ ]+"),
      Histologic_Grade = ifelse(Histologic_Grade == "GX", "G3", Histologic_Grade),
      
      Figo_Stage = {
        tmp <- str_extract(as.character(Figo_Stage), "(?<=\\().*(?=\\))")
        tmp <- str_remove(tmp, "^FIGO\\s+")
        tmp <- str_trim(tmp)
        ifelse(is.na(tmp), NA_character_, tmp)
      },
      
      Figo_Group = case_when(
        str_detect(Figo_Stage, "^I($|A|B|C)") ~ "I/II",
        str_detect(Figo_Stage, "^II($|A|B)") ~ "I/II",
        str_detect(Figo_Stage, "^III($|A|B|C)") ~ "III/IV",
        str_detect(Figo_Stage, "^IV($|A|B)") ~ "III/IV",
        TRUE ~ NA_character_
      ),
      
      Residual_Tumor = str_extract(as.character(Residual_Tumor), "^[^:]+") %>% str_trim(),
      
      RFS.time = suppressWarnings(as.numeric(as.character(RFS.time))) / 30,
      OS.time  = suppressWarnings(as.numeric(as.character(OS.time))) / 30,
      RFS.time = round(RFS.time, 2),
      OS.time  = round(OS.time, 2),
      
      RFS = suppressWarnings(as.integer(as.character(RFS))),
      OS  = suppressWarnings(as.integer(as.character(OS))),
      
      NTE.time   = suppressWarnings(as.numeric(as.character(NTE.days))) / 30,
      Last.time  = suppressWarnings(as.numeric(as.character(Last.days))) / 30,
      Death.time = suppressWarnings(as.numeric(as.character(Death.days))) / 30,
      
      NTE.time   = round(NTE.time, 2),
      Last.time  = round(Last.time, 2),
      Death.time = round(Death.time, 2),
      
      Death.time = ifelse(OS == 1L & is.na(Death.time), OS.time, Death.time),
      Last.time  = ifelse(OS == 0L & is.na(Last.time),  OS.time, Last.time),
      
      NTE_event = str_to_lower(str_trim(as.character(NTE_flag))) %in% c("1", "yes", "true", "y") | !is.na(NTE.time),
      
      PFS = ifelse(NTE_event | OS == 1L, 1L, 0L),
      
      PFS_event_time = {
        tt <- pmin(
          ifelse(NTE_event, NTE.time, NA_real_),
          ifelse(OS == 1L, Death.time, NA_real_),
          na.rm = TRUE
        )
        tt[is.infinite(tt)] <- NA_real_
        tt
      },
      
      PFS.time = dplyr::case_when(
        PFS == 1L ~ PFS_event_time,
        PFS == 0L ~ Last.time
      )
    )
  
  dim(clin_df_filtered)
  summary(clin_df_filtered$RFS.time)
  summary(clin_df_filtered$OS.time)
  summary(clin_df_filtered$PFS.time)
}

##03 Merge supplementary clinical annotations####
if (T) {
  clin_df_combined <- clin_df_filtered %>%
    left_join(
      clin_df2_clean %>%
        select(Patient_ID, Myometrial_invasion_Specify, LVSI, Genomics_subtype),
      by = "Patient_ID"
    ) %>%
    mutate(
      Myometrial_Invasion = case_when(
        str_detect(str_to_lower(Myometrial_invasion_Specify), "under\\s*50") ~ "<50%",
        str_detect(str_to_lower(Myometrial_invasion_Specify), "50\\s*%\\s*or\\s*more|50\\s*or\\s*more") ~ ">=50%",
        is.na(Myometrial_invasion_Specify) ~ "Unknown",
        str_detect(str_to_lower(Myometrial_invasion_Specify), "not\\s*identified|not\\s*reported|unknown|na") ~ "Unknown",
        TRUE ~ "Unknown"
      ),
      Subtype = case_when(
        str_detect(str_to_lower(Genomics_subtype), "^cnv[_ -]?high$") ~ "UCEC_CN_HIGH",
        str_detect(str_to_lower(Genomics_subtype), "^cnv[_ -]?low$")  ~ "UCEC_CN_LOW",
        str_detect(str_to_lower(Genomics_subtype), "^msi[-_ ]?h$|^msi$") ~ "UCEC_MSI",
        str_detect(str_to_lower(Genomics_subtype), "^pole$") ~ "UCEC_POLE",
        TRUE ~ "Unknown"
      ),
      Subtype = factor(
        Subtype,
        levels = c("UCEC_CN_HIGH", "UCEC_CN_LOW", "UCEC_MSI", "UCEC_POLE", "Unknown")
      )
    )
  
  write.csv(clin_df_combined, "clin_df_combined.csv", row.names = FALSE)
  dim(clin_df_combined)
}

## 04 Exclusion criteria####
if (TRUE) {
  step0_df <- clin_df_combined
  cat("Step0 总例数：", nrow(step0_df), "\n")
  
  step0_df <- step0_df %>%
    mutate(
      Prior_treatment_std = case_when(
        str_to_lower(as.character(Prior_Treatment)) %in% c("yes", "y", "1", "true") ~ "Yes",
        str_to_lower(as.character(Prior_Treatment)) %in% c("no", "n", "0", "false") ~ "No",
        TRUE ~ "Unknown"
      )
    )
  
  print(table(step0_df$Prior_Treatment))
  
  step1_df <- step0_df %>%
    filter(Prior_Treatment != "Yes")
  
  cat("Step1 剔除复发/治疗史后：", nrow(step1_df),
      "; 剔除：", nrow(step0_df) - nrow(step1_df), "\n")
  
  step2_df <- step1_df %>%
    filter(!is.na(PFS.time) & PFS.time >= 1)
  
  cat("Step2 剔除RFS.time缺失或<30天后：", nrow(step2_df),
      "; 剔除：", nrow(step1_df) - nrow(step2_df), "\n")
  
  step3_df <- step2_df %>%
    filter(
      !str_detect(
        as.character(Histological_Type),
        regex("\\bserous\\b|\\bmixed\\b|\\Clear\\b", ignore_case = TRUE)
      )
    )
  
  cat("Step3 剔除浆液型/混合型后：", nrow(step3_df),
      "; 剔除：", nrow(step2_df) - nrow(step3_df), "\n")
  
  step4_df <- step3_df %>%
    filter(!is.na(Figo_Group) & Figo_Group == "I/II")
  
  cat("Step4 剔除FIGO III/IV后：", nrow(step4_df),
      "; 剔除：", nrow(step3_df) - nrow(step4_df), "\n")
  
  # 保留你原脚本逻辑：这里仍然从 step2_df 开始
  final_df <- step4_df %>%
    filter(
      !is.na(Age) & Age != "",
      !is.na(Histologic_Grade) & Histologic_Grade != "",
      !is.na(Figo_Stage) & Figo_Stage != ""
    )
  
  cat("Step5 剔除关键信息缺失后最终：", nrow(final_df),
      "; 剔除：", nrow(step2_df) - nrow(final_df), "\n")
  
  final_df_ <- final_df %>%
    dplyr::select(
      Patient_ID,
      Age,
      Histologic_Grade,
      Figo_Group,
      Tumor_Stage_Pathological,
      Tumor_Necrosis,
      LVSI,
      Subtype,
      Myometrial_Invasion,
      Radiation_Therapy,
      Residual_Tumor,
      RFS.time, RFS,
      OS.time, OS,
      PFS.time, PFS
    )
  
  print(table(final_df_$Histologic_Grade, useNA = "ifany"))
  print(table(final_df_$Figo_Group, useNA = "ifany"))
  
  write.csv(final_df_, "CPTAC_UCEC_clin_final_filtered_94.csv", row.names = FALSE)
  dim(final_df_)
}


## 05 WSI matching + baseline summary table####
if (T) {
  setwd(path_wsi_id)
  
  rt1 <- final_df_
  
  wsi_ids <- read_csv("CPTAC_pathology_ID.csv", show_col_types = FALSE)[[1]] |>
    as.character() |>
    str_trim() |>
    (\(x) x[!is.na(x) & x != ""])() |>
    unique()
  
  rt1 <- rt1 %>%
    mutate(Patient_ID = str_trim(as.character(Patient_ID))) %>%
    filter(Patient_ID %in% wsi_ids)
  
  dim(rt1)
  
  vars <- c(
    "Age", "Histologic_Grade", "Subtype", "Myometrial_Invasion",
    "Tumor_Stage_Pathological", "Tumor_Necrosis", "LVSI",
    "Radiation_Therapy", "Residual_Tumor"
  )
  
  tbl1_overall <- rt1 %>%
    select(any_of(vars)) %>%
    mutate(Age = suppressWarnings(as.numeric(Age))) %>%
    tbl_summary(
      statistic = list(
        all_continuous() ~ "{median} ({p25}, {p75})",
        all_categorical() ~ "{n} ({p}%)"
      ),
      missing = "ifany"
    ) %>%
    modify_header(label ~ "**Characteristic**") %>%
    bold_labels()
  
  gt_tbl <- as_gt(tbl1_overall)
  gt::gtsave(gt_tbl, filename = "Table1_Baseline_CPTAC.png", zoom = 2)
}

## 06 Transcriptome matrix processing####
if (T) {
  expr_df <- fread(trans_path, data.table = FALSE, check.names = FALSE)
  cat("[Trans] dim:", dim(expr_df), "\n")
  
  if ("Database_ID" %in% expr_df$Name) {
    gene_map <- tibble(
      gene_symbol = colnames(expr_df)[-1],
      ensembl_id  = as.character(expr_df[expr_df$Name == "Database_ID", -1, drop = TRUE])
    )
    saveRDS(gene_map, "CPTAC_UCEC_gene_symbol_to_ensembl.rds")
  }
  
  expr_df2 <- expr_df %>%
    filter(!(Name %in% c("Database_ID", "Patient_ID")))
  
  dim(expr_df2)
  sum(duplicated(expr_df2$Name))
  print(table(expr_df2$Name))
  
  sample_id <- expr_df2$Name
  
  expr_mat <- as.matrix(expr_df2[, -1, drop = FALSE])
  rownames(expr_mat) <- sample_id
  suppressWarnings(storage.mode(expr_mat) <- "numeric")
  
  cat("[Trans] expr_mat dim:", dim(expr_mat), "\n")
  cat("[Trans] NA count:", sum(is.na(expr_mat)), "\n")
  
  expr_mat_log <- log2(expr_mat + 1)
  
  saveRDS(expr_mat, "CPTAC_UCEC_expr_mat_samples_by_genes.rds")
  saveRDS(expr_mat_log, "CPTAC_UCEC_expr_mat_log2p1_samples_by_genes.rds")
}

## 07 Match transcriptome with final clinical cohort####
if (T) {
  common_id <- intersect(rownames(expr_mat_log), rt1$Patient_ID)
  cat("common_id:", length(common_id), "\n")
  
  common_id <- common_id[!is.na(common_id) & common_id != ""]
  common_id <- sort(common_id)
  
  expr_mat_log_keep <- expr_mat_log[common_id, , drop = FALSE]
  dim(expr_mat_log_keep)
  
  expr_mat_log_keep <- t(as.matrix(expr_mat_log_keep))
  dim(expr_mat_log_keep)
}
## 08 Align CPTAC transcriptome to pathway gene space####
if (T) {
  length(rownames(mat_all_pc))
  
  pc_genes <- unique(rownames(mat_z))
  length(pc_genes)
  
  keep_cols_2 <- intersect(rownames(mRNAexp_filter_pathway_), keep_cols)
  sum(is.na(keep_cols_2))
  
  expr_mat_log_keep_pc <- expr_mat_log_keep[keep_cols_2, , drop = FALSE]
  dim(expr_mat_log_keep_pc)
  
  expr_mat_log_keep_pc[1:3, 1:8]
}