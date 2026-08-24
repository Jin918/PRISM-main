msigdb_dir <- "/Users/jin/Desktop/UCEC-CPTAC-PRISM/DATA/pathway_236/Mdbsi_pathway_processing_raw"

gmt_files <- c(
  "h.all.v2025.1.Hs.symbols.gmt",
  "c2.cp.kegg_legacy.v2025.1.Hs.symbols.gmt"
  # ,"c6.all.v2025.1.Hs.symbols.gmt"
)

gmt_paths <- file.path(msigdb_dir, gmt_files)
names(gmt_paths) <- c("HALLMARK", "C2_KEGG_LEGACY")
gmt_paths

read_gmt_to_binary_matrix <- function(gmt_path) {
  
  stopifnot(file.exists(gmt_path))
  
  lines <- readLines(gmt_path, warn = FALSE)
  
  splitted <- strsplit(lines, "\t", fixed = TRUE)
  
  # pathway-gene list
  pathway <- vapply(splitted, `[`, character(1), 1)
  genes_list <- lapply(splitted, function(x) {
    if (length(x) >= 3) x[3:length(x)] else character(0)
  })
  
  df_long <- data.frame(
    pathway = rep(pathway, times = vapply(genes_list, length, integer(1))),
    gene    = unlist(genes_list, use.names = FALSE),
    stringsAsFactors = FALSE
  )
  
  df_long <- df_long[df_long$gene != "" & !is.na(df_long$gene), , drop = FALSE]
  df_long <- unique(df_long)
  
  pathways <- unique(df_long$pathway)
  genes <- sort(unique(df_long$gene))
  
  mat <- matrix(0L, nrow = length(pathways), ncol = length(genes),
                dimnames = list(pathways, genes))
  mat[cbind(match(df_long$pathway, pathways), match(df_long$gene, genes))] <- 1L
  
  return(mat)
}

mat_h  <- read_gmt_to_binary_matrix(gmt_paths["HALLMARK"])
mat_k  <- read_gmt_to_binary_matrix(gmt_paths["C2_KEGG_LEGACY"])
# mat_c6 <- read_gmt_to_binary_matrix(gmt_paths["C6"])

dim(mat_h)
dim(mat_k)
# dim(mat_c6)

## mitrix combined
merge_binary_mats_by_genes <- function(mats_named_list) {
  all_genes <- sort(unique(unlist(lapply(mats_named_list, colnames), use.names = FALSE)))
  
  mats_aligned <- lapply(names(mats_named_list), function(nm) {
    m <- mats_named_list[[nm]]
    
    m2 <- matrix(0L, nrow = nrow(m), ncol = length(all_genes),
                 dimnames = list(rownames(m), all_genes))
    m2[, colnames(m)] <- m
    
    rownames(m2) <- paste0(nm, ":", rownames(m2))
    m2
  })
  names(mats_aligned) <- names(mats_named_list)
  
  do.call(rbind, mats_aligned)
}

mat_all <- merge_binary_mats_by_genes(list(
  HALLMARK = mat_h,
  C2_KEGG_LEGACY = mat_k
  # ,C6 = mat_c6
))

dim(mat_all)
colnames(mat_all)
rownames(mat_all)

saveRDS(
  mat_all,
  file.path(msigdb_dir, "MSigDB_2sets_binary_236x7336.rds")
)

write.csv(
  mat_all,
  file.path(msigdb_dir, "MSigDB_2sets_binary_236x7336.csv"),
  row.names = TRUE
)


# ##filter-TCGA
# ## with 01_TCGA_data_processing.R
# length(rownames(mRNAexp_filter))
# pc_genes <- unique(rownames(mRNAexp_filter))
# keep_cols <- intersect(colnames(mat_all), pc_genes)
# sum(is.na(keep_cols))
# dim(mat_all) #236 7336
# mat_all_pc <- mat_all[, keep_cols, drop = FALSE]
# mRNAexp_filter_pathway <- mRNAexp_filter[keep_cols, , drop = FALSE]
# dim(mat_all_pc) #236 6292
# dim(mRNAexp_filter_pathway) #6292  306
# write.csv(mat_all_pc, file.path(msigdb_dir, "MSigDB_2sets_co_genes.csv"))
# 
# 
# ##filter-CPTAC
# length(rownames(expr_mat_log_keep))
# pc_genes <- unique(rownames(expr_mat_log_keep))
# length(pc_genes)
# keep_cols <- intersect(rownames(mRNAexp_filter_pathway), pc_genes)
# sum(is.na(keep_cols))
# dim(mat_all)
# mat_all_pc <- mat_all[, keep_cols, drop = FALSE]
# mRNAexp_filter_pathway_ <- expr_mat_log_keep[keep_cols, , drop = FALSE]
# dim(mat_all_pc) #6292
# dim(mRNAexp_filter_pathway_) #6292  306
# 
# mat_align <- mat_all_pc[, colSums(mat_all_pc) > 0, drop = FALSE]
# dim(mat_align)

