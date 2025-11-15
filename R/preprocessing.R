#' Preprocess RNA-seq Data
#'
#' Normalize and preprocess scRNA-seq data (supports Seurat and SingleCellExperiment)
#'
#' @param obj Seurat or SingleCellExperiment object containing RNA-seq data
#' @param n_hvgs Number of highly variable genes to select (default: 2000)
#' @param min_cells Minimum number of cells expressing a gene (default: 3)
#' @param min_features Minimum number of features per cell (default: 200)
#' @param normalization_method Normalization method: "LogNormalize" or "SCT" (default: "LogNormalize")
#' @param scale_factor Scale factor for normalization (default: 10000)
#' @return Processed object (same type as input)
#' @export
#' @importFrom Seurat NormalizeData
preprocess_rna <- function(obj,
                           n_hvgs = 2000,
                           min_cells = 3,
                           min_features = 200,
                           normalization_method = "LogNormalize",
                           scale_factor = 10000) {

  # Dispatch to appropriate preprocessing function
  if (is_seurat(obj)) {
    return(preprocess_rna_seurat(obj, n_hvgs, min_cells, min_features,
                                 normalization_method, scale_factor))
  } else if (is_sce(obj)) {
    return(preprocess_rna_sce(obj, n_hvgs, min_cells, min_features,
                              normalization_method, scale_factor))
  } else {
    stop("Object must be either Seurat or SingleCellExperiment")
  }
}


#' Preprocess RNA-seq Data (Seurat)
#'
#' @keywords internal
preprocess_rna_seurat <- function(seurat_obj,
                                  n_hvgs = 2000,
                                  min_cells = 3,
                                  min_features = 200,
                                  normalization_method = "LogNormalize",
                                  scale_factor = 10000) {

  # Quality control filtering
  message("Filtering cells and genes...")
  # Filter genes: keep genes expressed in at least min_cells
  gene_counts <- Matrix::rowSums(Seurat::GetAssayData(seurat_obj, slot = "counts") > 0)
  genes_keep <- names(gene_counts)[gene_counts >= min_cells]
  seurat_obj <- seurat_obj[genes_keep, ]

  # Filter cells: keep cells with at least min_features
  cell_features <- Matrix::colSums(Seurat::GetAssayData(seurat_obj, slot = "counts") > 0)
  cells_keep <- names(cell_features)[cell_features >= min_features]
  seurat_obj <- seurat_obj[, cells_keep]

  message(paste("Kept", ncol(seurat_obj), "cells and", nrow(seurat_obj), "genes"))

  # Normalization
  message(paste("Normalizing data using", normalization_method, "..."))
  if (normalization_method == "LogNormalize") {
    seurat_obj <- Seurat::NormalizeData(seurat_obj,
                                        normalization.method = "LogNormalize",
                                        scale.factor = scale_factor)
  } else if (normalization_method == "SCT") {
    if (!requireNamespace("sctransform", quietly = TRUE)) {
      stop("Please install sctransform package for SCT normalization")
    }
    seurat_obj <- Seurat::SCTransform(seurat_obj, variable.features.n = n_hvgs)
  } else {
    stop("Unsupported normalization method. Use 'LogNormalize' or 'SCT'")
  }

  # Find variable features
  if (normalization_method == "LogNormalize") {
    message(paste("Finding", n_hvgs, "highly variable genes..."))
    seurat_obj <- Seurat::FindVariableFeatures(seurat_obj,
                                               selection.method = "vst",
                                               nfeatures = n_hvgs)
  }

  # Scale data
  if (normalization_method == "LogNormalize") {
    message("Scaling data...")
    seurat_obj <- Seurat::ScaleData(seurat_obj,
                                    features = Seurat::VariableFeatures(seurat_obj))
  }

  message("Preprocessing complete!")
  return(seurat_obj)
}


#' Preprocess RNA-seq Data (SingleCellExperiment)
#'
#' @keywords internal
preprocess_rna_sce <- function(sce,
                               n_hvgs = 2000,
                               min_cells = 3,
                               min_features = 200,
                               normalization_method = "LogNormalize",
                               scale_factor = 10000) {

  if (!requireNamespace("scater", quietly = TRUE)) {
    stop("Please install scater package: BiocManager::install('scater')")
  }

  # Quality control filtering
  message("Filtering cells and genes...")

  # Filter genes: keep genes expressed in at least min_cells
  if ("counts" %in% SummarizedExperiment::assayNames(sce)) {
    gene_counts <- Matrix::rowSums(SummarizedExperiment::assay(sce, "counts") > 0)
    genes_keep <- gene_counts >= min_cells
    sce <- sce[genes_keep, ]

    # Filter cells: keep cells with at least min_features
    cell_features <- Matrix::colSums(SummarizedExperiment::assay(sce, "counts") > 0)
    cells_keep <- cell_features >= min_features
    sce <- sce[, cells_keep]

    message(paste("Kept", ncol(sce), "cells and", nrow(sce), "genes"))

    # Normalization
    message(paste("Normalizing data using", normalization_method, "..."))
    if (normalization_method == "LogNormalize") {
      # Calculate size factors
      sce <- scater::logNormCounts(sce, size_factors = NULL)

      # Find highly variable genes
      if (requireNamespace("scran", quietly = TRUE)) {
        message(paste("Finding", n_hvgs, "highly variable genes using scran..."))
        dec <- scran::modelGeneVar(sce)
        hvg <- scran::getTopHVGs(dec, n = n_hvgs)
        SummarizedExperiment::rowData(sce)$highly_variable <-
          rownames(sce) %in% hvg
      } else {
        warning("scran not installed. Cannot select highly variable genes. Install with: BiocManager::install('scran')")
      }
    } else {
      stop("SCT normalization not supported for SingleCellExperiment. Use 'LogNormalize'")
    }

  } else {
    stop("Counts assay not found in SingleCellExperiment")
  }

  message("Preprocessing complete!")
  return(sce)
}


#' Preprocess Protein (CITE-seq) Data
#'
#' Normalize and preprocess CITE-seq protein expression data
#'
#' @param seurat_obj Seurat object containing protein data
#' @param assay Name of the protein assay (default: "ADT" or "protein")
#' @param normalization_method Normalization method: "CLR" or "LogNormalize" (default: "CLR")
#' @return Processed Seurat object
#' @export
preprocess_protein <- function(seurat_obj,
                               assay = "ADT",
                               normalization_method = "CLR") {

  if (!assay %in% names(seurat_obj@assays)) {
    # Try alternative name
    if ("protein" %in% names(seurat_obj@assays)) {
      assay <- "protein"
    } else {
      stop(paste("Assay", assay, "not found. Available assays:", paste(names(seurat_obj@assays), collapse = ", ")))
    }
  }

  message(paste("Normalizing protein data using", normalization_method, "..."))

  Seurat::DefaultAssay(seurat_obj) <- assay

  if (normalization_method == "CLR") {
    seurat_obj <- Seurat::NormalizeData(seurat_obj,
                                        normalization.method = "CLR",
                                        margin = 2)  # Normalize across cells
  } else if (normalization_method == "LogNormalize") {
    seurat_obj <- Seurat::NormalizeData(seurat_obj,
                                        normalization.method = "LogNormalize")
  } else {
    stop("Unsupported normalization method for protein data")
  }

  # Scale protein data
  seurat_obj <- Seurat::ScaleData(seurat_obj)

  message("Protein preprocessing complete!")
  return(seurat_obj)
}


#' Preprocess ATAC-seq Data
#'
#' Normalize and preprocess ATAC-seq chromatin accessibility data
#'
#' @param seurat_obj Seurat object containing ATAC data
#' @param assay Name of the ATAC assay (default: "ATAC" or "peaks")
#' @param normalization_method Normalization method: "TFIDF" or "LogNormalize" (default: "TFIDF")
#' @return Processed Seurat object
#' @export
preprocess_atac <- function(seurat_obj,
                            assay = "ATAC",
                            normalization_method = "TFIDF") {

  if (!assay %in% names(seurat_obj@assays)) {
    # Try alternative name
    if ("peaks" %in% names(seurat_obj@assays)) {
      assay <- "peaks"
    } else {
      stop(paste("Assay", assay, "not found. Available assays:", paste(names(seurat_obj@assays), collapse = ", ")))
    }
  }

  message(paste("Normalizing ATAC data using", normalization_method, "..."))

  Seurat::DefaultAssay(seurat_obj) <- assay

  if (normalization_method == "TFIDF") {
    # TF-IDF normalization for ATAC-seq
    # This requires Signac package
    if (!requireNamespace("Signac", quietly = TRUE)) {
      warning("Signac package not available. Using LogNormalize instead.")
      seurat_obj <- Seurat::NormalizeData(seurat_obj,
                                          normalization.method = "LogNormalize")
    } else {
      seurat_obj <- Signac::RunTFIDF(seurat_obj)
    }
  } else if (normalization_method == "LogNormalize") {
    seurat_obj <- Seurat::NormalizeData(seurat_obj,
                                        normalization.method = "LogNormalize")
  } else {
    stop("Unsupported normalization method for ATAC data")
  }

  message("ATAC preprocessing complete!")
  return(seurat_obj)
}


#' Apply Batch Correction
#'
#' Correct for batch effects using Harmony or other methods
#'
#' @param seurat_obj Seurat object
#' @param batch_col Name of the column containing batch information
#' @param method Batch correction method: "harmony" or "combat" (default: "harmony")
#' @return Batch-corrected Seurat object
#' @export
batch_correction <- function(seurat_obj,
                             batch_col,
                             method = "harmony") {

  if (!batch_col %in% colnames(seurat_obj@meta.data)) {
    stop(paste("Batch column", batch_col, "not found in metadata"))
  }

  message(paste("Applying batch correction using", method, "..."))

  if (method == "harmony") {
    if (!requireNamespace("harmony", quietly = TRUE)) {
      stop("Please install harmony package: install.packages('harmony')")
    }

    # Run PCA first if not already done
    if (!"pca" %in% names(seurat_obj@reductions)) {
      seurat_obj <- Seurat::RunPCA(seurat_obj, verbose = FALSE)
    }

    seurat_obj <- harmony::RunHarmony(seurat_obj, group.by.vars = batch_col)

  } else if (method == "combat") {
    if (!requireNamespace("sva", quietly = TRUE)) {
      stop("Please install sva package from Bioconductor")
    }

    expr_matrix <- Seurat::GetAssayData(seurat_obj, slot = "data")
    batch <- seurat_obj@meta.data[[batch_col]]

    corrected <- sva::ComBat(dat = as.matrix(expr_matrix),
                            batch = batch,
                            mod = NULL)

    seurat_obj <- Seurat::SetAssayData(seurat_obj,
                                       slot = "data",
                                       new.data = corrected)
  } else {
    stop("Unsupported batch correction method")
  }

  message("Batch correction complete!")
  return(seurat_obj)
}
