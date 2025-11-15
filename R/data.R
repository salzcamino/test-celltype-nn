#' Load Single-Cell Data
#'
#' Load single-cell data from various formats including H5AD (AnnData), RDS (Seurat/SCE)
#'
#' @param file_path Path to the data file (.h5ad, .rds, or .h5seurat)
#' @param format Format of the file: "h5ad", "rds", "h5seurat", or "auto" (default: "auto")
#' @param return_type Return type: "seurat", "sce", or "auto" (default: "auto" returns native format)
#' @return A Seurat or SingleCellExperiment object containing the single-cell data
#' @export
#' @importFrom Seurat CreateSeuratObject
load_anndata <- function(file_path, format = "auto", return_type = "auto") {
  if (format == "auto") {
    format <- tools::file_ext(file_path)
  }

  obj <- NULL

  if (format %in% c("h5ad", "h5")) {
    # Use SeuratDisk to read H5AD files
    if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
      stop("Please install SeuratDisk to read H5AD files: devtools::install_github('mojaveazure/seurat-disk')")
    }
    obj <- SeuratDisk::LoadH5Seurat(file_path)
  } else if (format == "rds") {
    obj <- readRDS(file_path)
  } else if (format == "h5seurat") {
    if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
      stop("Please install SeuratDisk")
    }
    obj <- SeuratDisk::LoadH5Seurat(file_path)
  } else {
    stop("Unsupported format. Please use 'h5ad', 'rds', or 'h5seurat'")
  }

  # Convert if necessary
  if (return_type == "seurat" && is_sce(obj)) {
    obj <- sce_to_seurat(obj)
  } else if (return_type == "sce" && is_seurat(obj)) {
    obj <- seurat_to_sce(obj)
  }

  return(obj)
}


#' Check if Object is Seurat
#'
#' @param obj Object to check
#' @return Logical indicating if object is Seurat
#' @export
is_seurat <- function(obj) {
  return(inherits(obj, "Seurat"))
}


#' Check if Object is SingleCellExperiment
#'
#' @param obj Object to check
#' @return Logical indicating if object is SingleCellExperiment
#' @export
is_sce <- function(obj) {
  return(inherits(obj, "SingleCellExperiment"))
}


#' Convert Seurat to SingleCellExperiment
#'
#' @param seurat_obj Seurat object
#' @param assay Assay to convert (default: "RNA")
#' @return SingleCellExperiment object
#' @export
seurat_to_sce <- function(seurat_obj, assay = "RNA") {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("Please install SingleCellExperiment: BiocManager::install('SingleCellExperiment')")
  }

  # Get counts and data
  counts <- Seurat::GetAssayData(seurat_obj, assay = assay, slot = "counts")
  logcounts <- Seurat::GetAssayData(seurat_obj, assay = assay, slot = "data")

  # Create SCE
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts, logcounts = logcounts)
  )

  # Add metadata
  SummarizedExperiment::colData(sce) <- seurat_obj@meta.data

  # Add variable features if available
  if (length(Seurat::VariableFeatures(seurat_obj)) > 0) {
    SummarizedExperiment::rowData(sce)$highly_variable <-
      rownames(sce) %in% Seurat::VariableFeatures(seurat_obj)
  }

  message("Converted Seurat to SingleCellExperiment")
  return(sce)
}


#' Convert SingleCellExperiment to Seurat
#'
#' @param sce SingleCellExperiment object
#' @param assay_name Name for the Seurat assay (default: "RNA")
#' @return Seurat object
#' @export
sce_to_seurat <- function(sce, assay_name = "RNA") {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    stop("Please install Seurat")
  }
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    stop("Please install SingleCellExperiment")
  }

  # Get counts
  counts <- SummarizedExperiment::assay(sce, "counts")

  # Create Seurat object
  seurat_obj <- Seurat::CreateSeuratObject(
    counts = counts,
    assay = assay_name,
    meta.data = as.data.frame(SummarizedExperiment::colData(sce))
  )

  # Add logcounts if available
  if ("logcounts" %in% SummarizedExperiment::assayNames(sce)) {
    logcounts <- SummarizedExperiment::assay(sce, "logcounts")
    seurat_obj <- Seurat::SetAssayData(seurat_obj, slot = "data", new.data = logcounts)
  }

  # Set variable features if available
  if ("highly_variable" %in% colnames(SummarizedExperiment::rowData(sce))) {
    hvg <- rownames(sce)[SummarizedExperiment::rowData(sce)$highly_variable]
    Seurat::VariableFeatures(seurat_obj) <- hvg
  }

  message("Converted SingleCellExperiment to Seurat")
  return(seurat_obj)
}


#' Split Data into Train/Validation/Test Sets
#'
#' Perform stratified splitting of single-cell data based on cell type labels
#'
#' @param obj Seurat or SingleCellExperiment object containing the data
#' @param train_frac Fraction of data for training (default: 0.7)
#' @param val_frac Fraction of data for validation (default: 0.15)
#' @param test_frac Fraction of data for testing (default: 0.15)
#' @param cell_type_col Name of the column containing cell type labels (default: "cell_type")
#' @param seed Random seed for reproducibility (default: 42)
#' @return List with train, validation, and test objects (same type as input)
#' @export
split_data <- function(obj,
                       train_frac = 0.7,
                       val_frac = 0.15,
                       test_frac = 0.15,
                       cell_type_col = "cell_type",
                       seed = 42) {

  if (abs(train_frac + val_frac + test_frac - 1.0) > 1e-6) {
    stop("Fractions must sum to 1.0")
  }

  set.seed(seed)

  # Get metadata based on object type
  if (is_seurat(obj)) {
    metadata <- obj@meta.data
  } else if (is_sce(obj)) {
    metadata <- as.data.frame(SummarizedExperiment::colData(obj))
  } else {
    stop("Object must be either Seurat or SingleCellExperiment")
  }

  # Get cell types
  if (!cell_type_col %in% colnames(metadata)) {
    stop(paste("Column", cell_type_col, "not found in metadata"))
  }

  cell_types <- metadata[[cell_type_col]]
  unique_types <- unique(cell_types)

  train_indices <- c()
  val_indices <- c()
  test_indices <- c()

  # Stratified splitting for each cell type
  for (ct in unique_types) {
    ct_indices <- which(cell_types == ct)
    n_ct <- length(ct_indices)

    # Shuffle indices
    ct_indices <- sample(ct_indices)

    # Calculate split points
    n_train <- floor(n_ct * train_frac)
    n_val <- floor(n_ct * val_frac)

    train_indices <- c(train_indices, ct_indices[1:n_train])
    val_indices <- c(val_indices, ct_indices[(n_train + 1):(n_train + n_val)])
    test_indices <- c(test_indices, ct_indices[(n_train + n_val + 1):n_ct])
  }

  return(list(
    train = obj[, train_indices],
    validation = obj[, val_indices],
    test = obj[, test_indices]
  ))
}


#' Create Data Loaders for Training
#'
#' Prepare data matrices and labels for neural network training
#'
#' @param obj Seurat or SingleCellExperiment object containing the data
#' @param cell_type_col Name of the column containing cell type labels
#' @param assay Name of the assay to use (default: "RNA" for Seurat, "logcounts" for SCE)
#' @param slot Name of the slot to use for Seurat (default: "data")
#' @return List with features matrix and encoded labels
#' @export
create_dataloaders <- function(obj,
                               cell_type_col = "cell_type",
                               assay = NULL,
                               slot = "data") {

  # Get expression matrix based on object type
  if (is_seurat(obj)) {
    # Seurat object
    if (is.null(assay)) assay <- "RNA"

    if (assay %in% names(obj@assays)) {
      expr_matrix <- Seurat::GetAssayData(obj, assay = assay, slot = slot)
    } else {
      stop(paste("Assay", assay, "not found"))
    }

    # Get labels
    labels <- obj@meta.data[[cell_type_col]]

  } else if (is_sce(obj)) {
    # SingleCellExperiment object
    if (is.null(assay)) assay <- "logcounts"

    if (!assay %in% SummarizedExperiment::assayNames(obj)) {
      # Try to find a suitable assay
      available_assays <- SummarizedExperiment::assayNames(obj)
      if ("logcounts" %in% available_assays) {
        assay <- "logcounts"
      } else if ("counts" %in% available_assays) {
        assay <- "counts"
        warning("Using counts assay. Consider log-normalizing first.")
      } else {
        stop(paste("Assay", assay, "not found. Available assays:",
                   paste(available_assays, collapse = ", ")))
      }
    }

    expr_matrix <- SummarizedExperiment::assay(obj, assay)

    # Get labels
    metadata <- as.data.frame(SummarizedExperiment::colData(obj))
    if (!cell_type_col %in% colnames(metadata)) {
      stop(paste("Column", cell_type_col, "not found in colData"))
    }
    labels <- metadata[[cell_type_col]]

  } else {
    stop("Object must be either Seurat or SingleCellExperiment")
  }

  # Transpose to cells x genes
  features <- t(as.matrix(expr_matrix))

  # Encode labels
  label_encoder <- list()
  unique_labels <- sort(unique(labels))
  for (i in seq_along(unique_labels)) {
    label_encoder[[unique_labels[i]]] <- i - 1  # 0-indexed for keras
  }

  encoded_labels <- sapply(labels, function(x) label_encoder[[x]])

  return(list(
    features = features,
    labels = encoded_labels,
    label_encoder = label_encoder,
    n_classes = length(unique_labels),
    n_features = ncol(features)
  ))
}
