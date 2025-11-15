#' Load AnnData or Seurat Object
#'
#' Load single-cell data from various formats including H5AD (AnnData) and RDS (Seurat)
#'
#' @param file_path Path to the data file (.h5ad, .rds, or .h5seurat)
#' @param format Format of the file: "h5ad", "rds", or "h5seurat"
#' @return A Seurat object containing the single-cell data
#' @export
#' @importFrom Seurat CreateSeuratObject
load_anndata <- function(file_path, format = "auto") {
  if (format == "auto") {
    format <- tools::file_ext(file_path)
  }

  if (format %in% c("h5ad", "h5")) {
    # Use SeuratDisk to read H5AD files
    if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
      stop("Please install SeuratDisk to read H5AD files: devtools::install_github('mojaveazure/seurat-disk')")
    }
    seurat_obj <- SeuratDisk::LoadH5Seurat(file_path)
  } else if (format == "rds") {
    seurat_obj <- readRDS(file_path)
  } else if (format == "h5seurat") {
    if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
      stop("Please install SeuratDisk")
    }
    seurat_obj <- SeuratDisk::LoadH5Seurat(file_path)
  } else {
    stop("Unsupported format. Please use 'h5ad', 'rds', or 'h5seurat'")
  }

  return(seurat_obj)
}


#' Split Data into Train/Validation/Test Sets
#'
#' Perform stratified splitting of single-cell data based on cell type labels
#'
#' @param seurat_obj Seurat object containing the data
#' @param train_frac Fraction of data for training (default: 0.7)
#' @param val_frac Fraction of data for validation (default: 0.15)
#' @param test_frac Fraction of data for testing (default: 0.15)
#' @param cell_type_col Name of the column containing cell type labels (default: "cell_type")
#' @param seed Random seed for reproducibility (default: 42)
#' @return List with train, validation, and test Seurat objects
#' @export
split_data <- function(seurat_obj,
                       train_frac = 0.7,
                       val_frac = 0.15,
                       test_frac = 0.15,
                       cell_type_col = "cell_type",
                       seed = 42) {

  if (abs(train_frac + val_frac + test_frac - 1.0) > 1e-6) {
    stop("Fractions must sum to 1.0")
  }

  set.seed(seed)

  # Get cell types
  if (!cell_type_col %in% colnames(seurat_obj@meta.data)) {
    stop(paste("Column", cell_type_col, "not found in metadata"))
  }

  cell_types <- seurat_obj@meta.data[[cell_type_col]]
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
    train = seurat_obj[, train_indices],
    validation = seurat_obj[, val_indices],
    test = seurat_obj[, test_indices]
  ))
}


#' Create Data Loaders for Training
#'
#' Prepare data matrices and labels for neural network training
#'
#' @param seurat_obj Seurat object containing the data
#' @param cell_type_col Name of the column containing cell type labels
#' @param assay Name of the assay to use (default: "RNA")
#' @param slot Name of the slot to use (default: "data")
#' @return List with features matrix and encoded labels
#' @export
create_dataloaders <- function(seurat_obj,
                               cell_type_col = "cell_type",
                               assay = "RNA",
                               slot = "data") {

  # Get expression matrix
  if (assay %in% names(seurat_obj@assays)) {
    expr_matrix <- Seurat::GetAssayData(seurat_obj, assay = assay, slot = slot)
  } else {
    stop(paste("Assay", assay, "not found"))
  }

  # Transpose to cells x genes
  features <- t(as.matrix(expr_matrix))

  # Get labels
  labels <- seurat_obj@meta.data[[cell_type_col]]

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
