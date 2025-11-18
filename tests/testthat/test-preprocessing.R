# Tests for preprocessing functions

test_that("preprocess_rna_seurat normalizes data correctly", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)

  processed <- preprocess_rna_seurat(
    seurat_obj,
    normalize = TRUE,
    scale = FALSE,
    n_variable_features = 20
  )

  expect_s4_class(processed, "Seurat")
  expect_true("LogNormalize" %in% names(processed@commands))
})

test_that("preprocess_rna_seurat finds variable features", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)

  processed <- preprocess_rna_seurat(
    seurat_obj,
    n_variable_features = 20
  )

  var_features <- Seurat::VariableFeatures(processed)
  expect_equal(length(var_features), 20)
})

test_that("preprocess_rna_seurat can scale data", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)

  processed <- preprocess_rna_seurat(
    seurat_obj,
    normalize = TRUE,
    scale = TRUE
  )

  expect_true("ScaleData" %in% names(processed@commands))
})

test_that("preprocess_protein_seurat handles CLR normalization", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 30)

  # Simulate protein assay
  protein_counts <- matrix(
    rpois(100 * 20, lambda = 10),
    nrow = 20,
    ncol = 100
  )
  rownames(protein_counts) <- paste0("Protein_", 1:20)

  seurat_obj[["ADT"]] <- Seurat::CreateAssayObject(counts = protein_counts)

  processed <- preprocess_protein_seurat(
    seurat_obj,
    assay = "ADT",
    method = "CLR"
  )

  expect_s4_class(processed, "Seurat")
})

test_that("split_data creates correct proportions", {
  X <- create_test_matrix(100, 50)
  y <- create_test_labels(100, 3)

  result <- split_data(
    X, y,
    train_ratio = 0.7,
    val_ratio = 0.15,
    test_ratio = 0.15
  )

  expect_equal(nrow(result$X_train), 70)
  expect_equal(nrow(result$X_val), 15)
  expect_equal(nrow(result$X_test), 15)

  expect_equal(length(result$y_train), 70)
  expect_equal(length(result$y_val), 15)
  expect_equal(length(result$y_test), 15)
})

test_that("split_data preserves features", {
  X <- create_test_matrix(100, 50)
  y <- create_test_labels(100, 3)

  result <- split_data(X, y)

  expect_equal(ncol(result$X_train), 50)
  expect_equal(ncol(result$X_val), 50)
  expect_equal(ncol(result$X_test), 50)
})

test_that("prepare_data_for_training works with Seurat", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)

  result <- prepare_data_for_training(
    seurat_obj,
    label_column = "cell_type",
    features = NULL
  )

  expect_true(is.matrix(result$X))
  expect_true(is.factor(result$y))
  expect_equal(nrow(result$X), ncol(seurat_obj))
})

test_that("prepare_data_for_training subsets features", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)
  features <- paste0("Gene_", 1:20)

  result <- prepare_data_for_training(
    seurat_obj,
    label_column = "cell_type",
    features = features
  )

  expect_equal(ncol(result$X), 20)
})

test_that("encode_labels creates correct encoding", {
  labels <- factor(c("TypeA", "TypeB", "TypeC", "TypeA", "TypeB"))

  result <- encode_labels(labels)

  expect_equal(length(result$encoded), 5)
  expect_true(all(result$encoded %in% 0:2))
  expect_equal(length(result$label_map), 3)
})

test_that("encode_labels is reversible", {
  labels <- factor(c("TypeA", "TypeB", "TypeC", "TypeA"))

  result <- encode_labels(labels)

  # Decode
  decoded <- names(result$label_map)[match(result$encoded + 1, result$label_map)]

  expect_equal(decoded, as.character(labels))
})

test_that("preprocessing handles missing values", {
  X <- create_test_matrix(100, 50)
  X[1:5, 1:5] <- NA

  y <- create_test_labels(100, 3)

  # Should handle NAs gracefully or error informatively
  expect_error(
    split_data(X, y),
    NA  # Or expect specific error message
  )
})
