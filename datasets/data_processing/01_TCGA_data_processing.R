rm(list = ls())

library(pacman)
p_load(TCGAbiolinks,SummarizedExperiment,data.table,readxl,
       ggplot2,ggstatsplot,ggsci,ggplotify,ggrepel,RColorBrewer,
       patchwork,limma,edgeR,DESeq2,stringr,
       FactoMineR,factoextra,sva,readr,VennDiagram,grid,
       caret,ggdist,pheatmap,CBCgrps,tidyverse,randomForestSRC,igraph)
library(gtsummary)
library(gt)



path_data <-("/Users/jin/Desktop/UCEC-CPTAC-PRISM/DATA/clinic/TCGA_287/raw")
path_r <- ("/Users/jin/Desktop/UCEC-CPTAC-PRISM/DATA/clinic/TCGA_287")

options(stringsAsFactors=F)
options(scipen=100)

##01 TCGA Expression Matrix Download ####
if(F){
  setwd(path_data)
  # getGDCprojects()$project_id
  # pro <- "TCGA-UCEC"
  # 
  # query <- GDCquery(
  #   project = pro,
  #   data.category="Transcriptome Profiling",
  #   data.type ="Gene Expression Quantification", 
  #   workflow.type="STAR - Counts"
  # )
  # 
  # GDCdownload(query=query, files.per.chunk= 50, directory = ".")
  # GDCprepare(query,save=T,save.filename=paste0(pro,".transcriptome.Rdata"), directory='.')
  
  load("TCGA-UCEC.transcriptome.Rdata")
  
  names(assays(data))
  rowdata<-rowData(data)
  table(rowdata$gene_type)
  
  mrna <- data[rowdata$gene_type == "protein_coding",]
  mrna_count <- assay(mrna,"unstranded") # counts matrix
  mrna_fpkm <- assay(mrna,"fpkm_unstrand") # FPKM martix
  table(rowData(mrna)$gene_type)
  
  symbol_mrna <- rowData(mrna)$gene_name
  head(symbol_mrna)
  any(duplicated(symbol_mrna)) #TRUE
  
  ## counts
  mrna_count_symbol <- cbind(data.frame(symbol_mrna),as.data.frame(mrna_count))
  mrna_count_df <- mrna_count_symbol
  colnames(mrna_count_df)[1] <- "symbol"
  
  cnt_symbol_sum <- mrna_count_df %>%
    as_tibble() %>%
    filter(!is.na(symbol) & symbol != "") %>%
    group_by(symbol) %>%
    summarise(across(where(is.numeric), ~ sum(.x, na.rm = TRUE)), .groups = "drop")
  
  cnt_symbol_sum[1:4, 1:5]
  cnt_symbol_sum_df <- as.data.frame(cnt_symbol_sum, check.names = FALSE)
  rownames(cnt_symbol_sum_df) <- cnt_symbol_sum_df$symbol
  cnt_symbol_sum_df$symbol <- NULL
  dim(cnt_symbol_sum_df) #19938   589
  
  saveRDS(cnt_symbol_sum_df, file = "tcga_ucec_mrna_count_symbol.rds")
  
  ## FPKM
  mrna_fpkm_symbol <- cbind(data.frame(symbol_mrna),as.data.frame(mrna_fpkm))
  
  mrna_fpkm_symbol1 <- mrna_fpkm_symbol %>% 
    as_tibble() %>%
    mutate(symbol_mrna = trimws(as.character(symbol_mrna))) %>%
    filter(!is.na(symbol_mrna) & symbol_mrna != "") %>%
    group_by(symbol_mrna) %>%
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    as.data.frame()
  
  mrna_fpkm_symbol1[1:4, 1:5]
  rownames(mrna_fpkm_symbol1) <- mrna_fpkm_symbol1$symbol_mrna
  mrna_fpkm_symbol2 <- mrna_fpkm_symbol1[,-1]
  dim(mrna_fpkm_symbol2) #19938   589
  
  # mRNAexp <- mrna_fpkm_symbol2
  # mRNAexp = mRNAexp[rowSums(mRNAexp)>0,]
  # nrow(mRNAexp)
  
  ## Filter
  # mRNAexp = mRNAexp[apply(mRNAexp, 1, function(x) sum(x > 0.1) > 0.1*ncol(mRNAexp)), ]
  #17933
  # 至少 10 个样本有表达（FPKM>0.1）
  # min_n <- 30
  # thr   <- 0.1
  # mRNAexp <- mRNAexp[rowSums(mRNAexp > thr, na.rm=TRUE) >= min_n, ]
  #18157
  # nrow(mRNAexp)
  # mrna_fpkm_symbol2 <- mRNAexp
  
  saveRDS(mrna_fpkm_symbol2, file = "tcga_ucec_mrna_fpkm_symbol.rds")
}

# exp_TCGA_Count <- readRDS(file="tcga_ucec_mrna_count_symbol.rds")
# exp_TCGA_fpkm <- readRDS(file="tcga_ucec_mrna_fpkm_symbol.rds")


##02 Inclusion and Exclusion Criteria ####
##Inclusion 
if(F){
  setwd(path_data)
  # query1 <- GDCquery(
  #   project = "TCGA-UCEC",
  #   data.category = "Clinical",
  #   data.type = "Clinical Supplement",
  #   data.format = "bcr xml"
  # )
  # 
  # GDCdownload(query1)
  # 
  # saveRDS(query1, file = "query1_TCGA_UCEC_Clinic.rds")
  query1 <- readRDS(file='query1_TCGA_UCEC_Clinic.rds')
  
  clinical.patient.xml <- GDCprepare_clinic(query1, clinical.info = "patient")
  cat("TCGA clinic matrix dimensions:", dim(clinical.patient.xml), "\n") #548 74 
  clinical.drug.xml <- GDCprepare_clinic(query1, clinical.info = "drug")
  cat("TCGA drug matrix dimensions:", dim(clinical.drug.xml), "\n") #479 24
  clinical.radiation.xml <- GDCprepare_clinic(query1, clinical.info = "radiation")
  cat("TCGA radiation matrix dimensions:", dim(clinical.radiation.xml), "\n") #304 20 
  clinical.followup.xml <- GDCprepare_clinic(query1, clinical.info = "follow_up")
  cat("TCGA followup matrix dimensions:", dim(clinical.followup.xml), "\n") #752 38
  clinical.newtumorevent.xml <- GDCprepare_clinic(query1, clinical.info = "new_tumor_event")
  cat("TCGA newtumorevent matrix dimensions:", dim(clinical.newtumorevent.xml), "\n") #396 13 
  table(clinical.newtumorevent.xml$new_neoplasm_event_type)
  ucec_gdc_clinic_data <- read_tsv("ucec_tcga_gdc_clinical_data.tsv")
  ucec_gdc_clinic_data <- as.data.frame(ucec_gdc_clinic_data)
  cat("TCGA gdc clinic matrix dimensions:", dim(ucec_gdc_clinic_data), "\n") #548 36
  ucec_cbioprotal <- read_tsv("ucec_tcga_pan_can_atlas_2018_clinical_data.tsv")
  ucec_cbioprotal <- as.data.frame(ucec_cbioprotal)
  cat("TCGA cbioprotal matrix dimensions:", dim(ucec_cbioprotal), "\n") #529 63
  TCGA_cdr <- read_xlsx("TCGA-CDR-UCEC.xlsx")
  TCGA_cdr <- as.data.frame(TCGA_cdr)
  cat("TCGA_cdr dimensions:", dim(TCGA_cdr), "\n") #548 19 
  
  # gdc clinic
  colnames(ucec_gdc_clinic_data)
  clinic_gdc_548 <- ucec_gdc_clinic_data %>%
    dplyr::rename(
      ID = `Patient ID`,   
      prior_treatment = `Prior Treatment`,
      primary_diagnosis = `Primary Diagnosis`,
      sample_type = `Sample Type`,
      Age = `Diagnosis Age`,
      FIGO_Stage = `FIGO Stage`,
      OS.time = `Overall Survival (Months)`,
      OS = `Overall Survival Status`,
      DFS.time = `Disease Free (Months)`,
      DFS = `Disease Free Status`
    ) %>%
    dplyr::select(ID, prior_treatment, primary_diagnosis, sample_type, Age, FIGO_Stage, OS.time, OS,DFS.time,DFS)
  dim(clinic_gdc_548) #548  10
  colnames(clinic_gdc_548)
  
  ## patient
  clinical_patient_548 <- clinical.patient.xml %>%
    dplyr::rename(
      ID = bcr_patient_barcode,
      Histological_Type = histological_type,
      Histologic_Grade = neoplasm_histologic_grade,
      Residual_Tumor = residual_tumor,
      neoadjuvant_treatment = history_of_neoadjuvant_treatment,
      myometrial_invasion = pct_tumor_invasion
    ) %>%
    dplyr::mutate(
      Histological_Type = ifelse(is.na(Histological_Type) | Histological_Type == "", "Unknown", as.character(Histological_Type)),
      Histologic_Grade  = ifelse(is.na(Histologic_Grade)  | Histologic_Grade  == "", "Unknown", as.character(Histologic_Grade)),
      Residual_Tumor    = ifelse(is.na(Residual_Tumor)    | Residual_Tumor    == "", "RX",      as.character(Residual_Tumor)),
      neoadjuvant_treatment = ifelse(is.na(neoadjuvant_treatment) | neoadjuvant_treatment == "", "Unknown", as.character(neoadjuvant_treatment)),
      myometrial_invasion = suppressWarnings(as.numeric(myometrial_invasion)),
      mi_50 = dplyr::case_when(
        is.na(myometrial_invasion) ~ "Unknown",
        myometrial_invasion >= 50  ~ ">=50%",
        TRUE                       ~ "<50%"
      )
    ) %>%
    dplyr::select(ID, Histological_Type, Histologic_Grade, Residual_Tumor,
                  neoadjuvant_treatment, myometrial_invasion, mi_50)
  dim(clinical_patient_548)  #548   7
  # TCGA-AJ-A4ZG
  
  ## drug
  clinical_drug_201 <- clinical.drug.xml %>%
    dplyr::select(bcr_patient_barcode, therapy_types) %>%
    dplyr::rename(ID = bcr_patient_barcode) %>%
    dplyr::mutate(Chemotherapy = ifelse(is.na(therapy_types) | therapy_types == "" | therapy_types == "Unknown",
                                        "Unknown",
                                        ifelse(grepl("Chemotherapy", therapy_types), "YES", "NO"))) %>%
    # 针对同一患者ID进行分组并合并
    group_by(ID) %>%
    summarise(Chemotherapy = case_when(
      any(Chemotherapy == "YES") ~ "YES",
      any(Chemotherapy == "Unknown") ~ "Unknown",
      TRUE ~ "NO" # 其他情况（NO）返回 NO
    ), .groups = "drop")
  dim(clinical_drug_201) #201   2
  
  ## TCGA-cdr 数据整理
  cdr_548 <- TCGA_cdr %>%
    dplyr::select(-1) %>%                       
    dplyr::rename(ID = bcr_patient_barcode) %>%   
    dplyr::mutate(
      ID = substr(as.character(ID), 1, 12)        # 统一为12位病人条码
    ) %>%
    # 把 #N/A / 空字符串转 NA，再把数值列转 numeric
    dplyr::mutate(
      across(everything(), ~ {
        x <- as.character(.x)
        x[x %in% c("#N/A", "N/A", "")] <- NA
        x
      })
    ) %>%
    dplyr::mutate(
      type = as.character(type),
      across(-c(ID, type), ~ suppressWarnings(as.numeric(.x)))
    ) %>%
    dplyr::select(
      ID, PFI.1, PFI.time.1, PFI.2, PFI.time.2, PFS, PFS.time
    )
  
  cat("cdr_548 dimensions:", dim(cdr_548), "\n") #548 7 
  
  
  # 整理cbioportal数据
  clinic_cbioprotal_529 <- ucec_cbioprotal %>%
    dplyr::rename(
      ID = `Patient ID`,
      Radiation_Therapy = `Radiation Therapy`,
      Subtype = Subtype,
      TMB = `TMB (nonsynonymous)`,
      Aneuploidy_Score = `Aneuploidy Score`,
      Fraction_Genome_Altered = `Fraction Genome Altered`
      # ,PFS.time = `Progress Free Survival (Months)`,
      # PFS = `Progression Free Status`
    ) %>%
    dplyr::mutate(
      Radiation_Therapy = ifelse(is.na(Radiation_Therapy) | Radiation_Therapy == "",
                                 "Unknown",
                                 as.character(Radiation_Therapy)),
      Subtype = ifelse(is.na(Subtype) | Subtype == "", "Unknown", as.character(Subtype))
    ) %>%
    dplyr::select(ID, Radiation_Therapy, Subtype, TMB, Aneuploidy_Score, Fraction_Genome_Altered)
  dim(clinic_cbioprotal_529) #529   6
  colnames(clinic_cbioprotal_529)
  
  table(cdr_548$PFI.1)
  table(cdr_548$PFI.2)
  table(cdr_548$PFS)
  table(clinic_cbioprotal_529$PFS)
  ##newtumorevent
  newtumorevent_1row <- clinical.newtumorevent.xml %>%
    dplyr::rename(ID = bcr_patient_barcode) %>%
    dplyr::mutate(
      ID = as.character(ID),
      new_neoplasm_event_type = dplyr::na_if(as.character(new_neoplasm_event_type), "")
    ) %>%
    dplyr::group_by(ID) %>%
    dplyr::summarise(
      new_neoplasm_event_type = ifelse(
        all(is.na(new_neoplasm_event_type)),
        "None/Unknown",
        paste(sort(unique(na.omit(new_neoplasm_event_type))), collapse = "; ")
      ),
      n_records = dplyr::n(),
      .groups = "drop"
    )
  
  # 检查：ID 是否唯一
  stopifnot(!any(duplicated(newtumorevent_1row$ID)))
  
  
  ## 合并所有数据
  final_df <- clinic_gdc_548 %>%
    left_join(clinical_patient_548, by = "ID") %>%
    left_join(clinical_drug_201, by = "ID") %>%
    left_join(clinic_cbioprotal_529, by = "ID") %>%
    left_join(cdr_548, by = "ID") %>%
    left_join(newtumorevent_1row, by = "ID") %>%
    mutate(
      Histological_Type = ifelse(is.na(Histological_Type) | Histological_Type == "", "Unknown", Histological_Type),
      Histologic_Grade  = ifelse(is.na(Histologic_Grade)  | Histologic_Grade  == "", "Unknown", Histologic_Grade),
      Residual_Tumor    = ifelse(is.na(Residual_Tumor)    | Residual_Tumor    == "", "Unknown", Residual_Tumor),
      Chemotherapy      = ifelse(is.na(Chemotherapy)      | Chemotherapy      == "", "Unknown", Chemotherapy),
      Radiation_Therapy = ifelse(is.na(Radiation_Therapy) | Radiation_Therapy == "", "Unknown", Radiation_Therapy),
      Subtype           = ifelse(is.na(Subtype)           | Subtype           == "", "Unknown", Subtype),
      new_neoplasm_event_type= ifelse(is.na(new_neoplasm_event_type)|new_neoplasm_event_type== "", "Unknown", new_neoplasm_event_type)
    )
  
  dim(final_df) #548  30
  colnames(final_df)
  saveRDS(final_df, file = "TCGA_UCEC_Clinic_ALL.rds")
}
final_df <- readRDS(file.path(path_data, "TCGA_UCEC_Clinic_ALL.rds"))

##Exclusion for PFS
if(T){
  # Step 1：基线样本定义（只剔除 prior_treatment 为 TRUE；NA 不剔除）
  step1_df <- final_df %>%
    mutate(
      prior_treatment = toupper(as.character(prior_treatment)),
      sample_type = as.character(sample_type)
    ) %>%
    filter(sample_type != "Recurrent Solid Tumor") %>%
    filter(is.na(prior_treatment) | prior_treatment != "TRUE")
  
  cat("After baseline filter:", nrow(step1_df), 
      " removed:", nrow(final_df) - nrow(step1_df), "\n")
  
  # Step2：剔除PFS.time为空或PFS.time<30天的样本
  step2_df <- step1_df%>%
    # filter(!is.na(OS.time) & OS.time != "" & as.numeric(OS.time) >= 1) %>%
    filter(!is.na(PFS.time) & PFS.time != "" & as.numeric(PFS.time) >= 30)
  
  cat("After PFS.time filter：", nrow(step2_df), "; removed：", nrow(step1_df) - nrow(step2_df), "\n")
  
  # Step3：剔除病理分型为浆液型和混合型的样本
  table(step2_df$Histological_Type)
  exclude_types <- c("Serous endometrial adenocarcinoma","Mixed serous and endometrioid")

  step3_df <- step2_df %>%
     mutate(Histological_Type = as.character(Histological_Type)) %>%
     filter(!Histological_Type %in% exclude_types)
   cat("After Histological_Type filter：", nrow(step3_df), "; removed：", nrow(step2_df) - nrow(step3_df), "\n")
   
  # Step4：剔除FIGO分期为 III/IV期的样本
   step4_df <- step3_df %>%
     mutate(
       FIGO_group = ifelse(FIGO_Stage %in% c("Stage I","Stage IA","Stage IB","Stage IC",
                                             "Stage II","Stage IIA","Stage IIB"),
                           "I/II", "III/IV")
     ) %>%
     filter(FIGO_group == "I/II")
  
  cat("After FIGO filter：", nrow(step4_df), "; removed：", nrow(step3_df) - nrow(step4_df), "\n")
  
  # Step5：剔除Age、Histologic_grade和FIGO_stage缺失样本
  step5_df <- step4_df %>%
    mutate(
      Histologic_Grade = dplyr::recode(Histologic_Grade, `High Grade` = "G3")
    ) %>%
    filter(
      !is.na(Age) & Age != "",
      !is.na(Histologic_Grade) & Histologic_Grade != "",
      !is.na(FIGO_Stage) & FIGO_Stage != ""
    )
  cat("fter NA filter：", nrow(step5_df), "; removed：", nrow(step2_df) - nrow(step5_df), "\n")
  dim(step5_df)
  
  # Step6: 针对OS/PFS变量的格式转换
  step6_df <- step5_df %>%
    mutate(across(
      .cols = c(PFS,OS),
      .fns = ~ as.numeric(
        str_extract(
          string = as.character(.x),
          pattern = "^\\d+(?=:|$)" 
        )
      )
    )) %>%
    filter(
      !is.na(PFS) &
        !is.na(PFS.time)
    )
  
  # Step6：变量全局梳理
  step6_df <- as.data.frame(step6_df)
  colnames(step6_df)
  filtered_arrage_df <- step6_df %>% 
    dplyr::select(ID, Age,FIGO_Group=FIGO_group, Histologic_Grade,Subtype,
                  TMB,Aneuploidy_Score,Fraction_Genome_Altered,
                  Radiation_Therapy,Residual_Tumor,
                  Myometrial_Invasion = mi_50,
                  PFS,PFS.time,OS,OS.time
    ) %>%
    dplyr::mutate(
      PFS.time      = suppressWarnings(as.numeric(as.character(PFS.time))),
      TMB       = suppressWarnings(as.numeric(as.character(TMB))),
      PFS.time = PFS.time / 30,
      TMB    = log(TMB+1)
    )
  dim(filtered_arrage_df) #312  15
  colnames(filtered_arrage_df)
  sum(duplicated(filtered_arrage_df$ID))
  
  for(i in names(filtered_arrage_df)){
    if(is.character(filtered_arrage_df[,i])){
      filtered_arrage_df[,i]=as.factor(filtered_arrage_df[,i])
    }
  }
  setwd(path_data)
  write.csv(filtered_arrage_df,"EC_early_Clinic_PFS_312.csv",row.names = FALSE)
} 

EC_early_df_312 <- read.csv("EC_early_Clinic_PFS_312.csv",header = TRUE,check.names = FALSE)

##03 Transcript data cleaning####
if(T){
  ## read data
  exp_TCGA_fpkm <- readRDS(file="tcga_ucec_mrna_fpkm_symbol.rds")
  dim(exp_TCGA_fpkm)
  
  ## Normal & Tumor grouping
  group_list <- str_sub(colnames(exp_TCGA_fpkm), 14, 16)
  table(group_list) #543   8   2   1  35
  
  valid_indices <- group_list %in% c("01A", "11A")
  exp_TCGA_fpkm <- exp_TCGA_fpkm[, valid_indices]
  group_list <- group_list[valid_indices]
  table(group_list) #543  35
  group_list <- ifelse(group_list == "11A", "normal", "tumor")
  table(group_list) #35 543 
  
  ## ID Rename
  dim(exp_TCGA_fpkm)
  
  old_names <- colnames(exp_TCGA_fpkm)
  old_group <- group_list
  
  new_names <- ifelse(old_group == "tumor", substr(old_names, 1, 12), old_names)
  colnames(exp_TCGA_fpkm) <- new_names
  
  dup_names <- unique(new_names[duplicated(new_names)]) 
  cat("rename ID:",dup_names, "\n") #TCGA-BK-A139 TCGA-BK-A26L TCGA-BK-A0CC TCGA-BK-A0CA 
  if(length(dup_names) > 0){
    unique_samples <- unique(new_names)
    agg_matrix <- matrix(NA, nrow = nrow(exp_TCGA_fpkm), ncol = length(unique_samples))
    rownames(agg_matrix) <- rownames(exp_TCGA_fpkm)
    colnames(agg_matrix) <- unique_samples
    for(i in seq_along(unique_samples)){
      idx <- which(new_names == unique_samples[i])
      if(length(idx) == 1){
        agg_matrix[, i] <- exp_TCGA_fpkm[, idx]
      } else {
        agg_matrix[, i] <- rowMeans(exp_TCGA_fpkm[, idx])
      }
    }
    
    exp_TCGA_fpkm <- agg_matrix
    group_list <- old_group[match(unique_samples, new_names)]
  }
  cat("TCGA fpkm matrix dimensions:", dim(exp_TCGA_fpkm), "\n")
  cat("group_list:",table(group_list),"\n") #35    539
  
  clin_ids <- unique(EC_early_df_312$ID)
  expr_ids <- colnames(exp_TCGA_fpkm)
  common_ids <- intersect(expr_ids, clin_ids);length(common_ids) #306
  exp_TCGA_fpkm_306 <- exp_TCGA_fpkm[, common_ids, drop = FALSE]
  EC_early_df_306 <- EC_early_df_312[match(common_ids, EC_early_df_312$ID), , drop = FALSE]
}

## 04 FPKM filtering & Pathway matrix arrangement ####
## with 00_MSigDB_data_processing.R
if(T){
  mRNAexp <- exp_TCGA_fpkm_306
  mRNAexp = mRNAexp[rowSums(mRNAexp)>0,]
  nrow(mRNAexp) #19448
  
  ## 基因至少在20%的样本中有表达，根据实际情况调整
  mRNAexp_filter = mRNAexp[apply(mRNAexp, 1, function(x) sum(x > 0.1) > 0.2*ncol(mRNAexp)), ]
  dim(mRNAexp_filter) #15973
  path_ids <- read.csv("WSI_ID.csv",stringsAsFactors = FALSE)$ID
  
  ##get 'mRNAexp_filter_pathway' from 00_MSigDB_data_processing.R
  
  ## log
  mRNAexp_filter_pathway_log <- log2(mRNAexp_filter_pathway + 1)
  
  mat <- mRNAexp_filter_pathway_log;dim(mat)  # genes x samples 6292  306
  
  gene_sd <- apply(mat, 1, sd, na.rm = TRUE)
  keep_gene <- !is.na(gene_sd) & gene_sd > 0
  mat <- mat[keep_gene, , drop = FALSE];dim(mat)
  
  gene_mu <- rowMeans(mat, na.rm = TRUE)
  gene_sd <- apply(mat, 1, sd, na.rm = TRUE)
  gene_sd[is.na(gene_sd) | gene_sd == 0] <- 1
  
  mat_z <- sweep(mat, 1, gene_mu, "-")
  mat_z <- sweep(mat_z, 1, gene_sd, "/")
  
  cat("row-mean range:", range(rowMeans(mat_z, na.rm=TRUE)), "\n")
  cat("row-sd range:", range(apply(mat_z, 1, sd, na.rm=TRUE)), "\n")
  
  write.csv(mat_z,"TCGA_UCEC_log_FPKM_pathway_normolized_306.csv")
}


## 05 Baseline Characteristics Table ####
## with 02_CPTAC_data_processing.R
wsi_co_ids <- read_csv("EC_Clinic_PFS_287_ID.csv")
common_ids <- intersect(wsi_co_ids$ID, EC_early_df_306$ID);length(common_ids)
exp_TCGA_fpkm_287 <- exp_TCGA_fpkm_306[, common_ids, drop = FALSE];dim(exp_TCGA_fpkm_287)
EC_early_df_287 <- EC_early_df_306[match(common_ids, EC_early_df_306$ID), , drop = FALSE];dim(EC_early_df_287)
write.csv(EC_early_df_287, "EC_Clinic_PFS_287.csv",row.names = FALSE)
rt <- EC_early_df_287 %>%
  mutate(
    PFS.time = as.numeric(as.character(PFS.time)),
    PFS      = as.integer(as.character(PFS))
  )

tcga_base <- rt %>%
  dplyr::select(-TMB, -Aneuploidy_Score, -Fraction_Genome_Altered) %>%
  transmute(
    cohort = "TCGA-UCEC",
    Age = Age,
    Histologic_Grade = Histologic_Grade,
    Subtype =Subtype,
    Myometrial_Invasion = Myometrial_Invasion,   
    Radiation_Therapy = Radiation_Therapy,
    Residual_Tumor = Residual_Tumor
  )

cptac_base <- rt1 %>%
  transmute(
    cohort = "CPTAC-UCEC",
    Age = Age,
    Histologic_Grade = Histologic_Grade,
    Subtype =Subtype,
    Myometrial_Invasion = Myometrial_Invasion,
    Radiation_Therapy = Radiation_Therapy,
    Residual_Tumor = Residual_Tumor
  )

tcga_base <- tcga_base %>%
  mutate(Age = as.numeric(Age))

cptac_base <- cptac_base %>%
  mutate(Age = readr::parse_number(as.character(Age))) 

base_df <- bind_rows(tcga_base, cptac_base) %>%
  mutate(
    cohort = factor(cohort, levels = c("TCGA-UCEC", "CPTAC-UCEC")),
    Histologic_Grade = factor(Histologic_Grade, levels = c("G1", "G2", "G3")),
    Subtype = factor(Subtype, levels = c("UCEC_CN_HIGH", "UCEC_CN_LOW", "UCEC_MSI","UCEC_POLE","Unknown")),
    Myometrial_Invasion = factor(Myometrial_Invasion, levels = c("<50%", ">=50%", "Unknown")),
    Radiation_Therapy = factor(Radiation_Therapy, levels = c("No", "Yes", "Unknown")),
    Residual_Tumor = factor(Residual_Tumor, levels = c("R0", "R1", "RX"))
  )


tbl1 <- base_df %>%
  tbl_summary(
    by = cohort,
    statistic = list(
      all_continuous() ~ "{median} ({p25}, {p75})",
      all_categorical() ~ "{n} ({p}%)"
    ),
    digits = all_continuous() ~ 1,
    missing = "ifany"
  ) %>%
  add_overall(last = TRUE)
tbl1

setwd(path_r)
write_csv(as_tibble(tbl1), "Table1_Baseline_TCGA_vs_CPTAC.csv")
gt_tbl <- as_gt(tbl1)
gt::gtsave(gt_tbl, filename = "Table1_Baseline_TCGA_vs_CPTAC.png", zoom = 2)

