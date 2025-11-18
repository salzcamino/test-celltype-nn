# Helper functions and fixtures for testthat

#' Create a simple Seurat object for testing
#'
#' @param n_cells Number of cells (default: 100)
#' @param n_genes Number of genes (default: 50)
#' @param n_celltypes Number of cell types (default: 3)
#' @return A Seurat object
create_test_seurat <- function(n_cells = 100, n_genes = 50, n_celltypes = 3) {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    skip("Seurat not installed")
  }

  # Create random expression matrix
  set.seed(42)
  counts <- matrix(
    rpois(n_cells * n_genes, lambda = 5),
    nrow = n_genes,
    ncol = n_cells
  )
  rownames(counts) <- paste0("Gene_", seq_len(n_genes))
  colnames(counts) <- paste0("Cell_", seq_len(n_cells))

  # Create Seurat object
  seurat_obj <- Seurat::CreateSeuratObject(
    counts = counts,
    project = "TestProject"
  )

  # Add cell type labels
  cell_types <- paste0("Type", sample(1:n_celltypes, n_cells, replace = TRUE))
  seurat_obj$cell_type <- cell_types

  return(seurat_obj)
}

#' Create a simple matrix for testing
#'
#' @param n_rows Number of rows
#' @param n_cols Number of columns
#' @return A numeric matrix
create_test_matrix <- function(n_rows = 100, n_cols = 50) {
  set.seed(42)
  matrix(rnorm(n_rows * n_cols), nrow = n_rows, ncol = n_cols)
}

#' Create test labels
#'
#' @param n Number of labels
#' @param n_classes Number of classes
#' @return A factor vector
create_test_labels <- function(n = 100, n_classes = 3) {
  set.seed(42)
  factor(sample(1:n_classes, n, replace = TRUE))
}

#' Skip if Keras/TensorFlow not available
skip_if_no_keras <- function() {
  if (!requireNamespace("keras", quietly = TRUE)) {
    skip("keras not installed")
  }

  tryCatch({
    keras::backend()
  }, error = function(e) {
    skip("TensorFlow backend not available")
  })
}

#' Skip if Seurat not available
skip_if_no_seurat <- function() {
  if (!requireNamespace("Seurat", quietly = TRUE)) {
    skip("Seurat not installed")
  }
}

#' Check if two numeric values are approximately equal
#'
#' @param x First value
#' @param y Second value
#' @param tolerance Tolerance for comparison
#' @return TRUE if approximately equal
approx_equal <- function(x, y, tolerance = 1e-7) {
  abs(x - y) < tolerance
}
