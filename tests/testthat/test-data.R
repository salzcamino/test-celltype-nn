# Tests for data handling functions

test_that("load_seurat_data loads Seurat object", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat()

  # Save to temp file
  temp_file <- tempfile(fileext = ".rds")
  saveRDS(seurat_obj, temp_file)

  # Load it back
  loaded <- load_seurat_data(temp_file)

  expect_s4_class(loaded, "Seurat")
  unlink(temp_file)
})

test_that("extract_expression_matrix returns correct format", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)

  expr_matrix <- extract_expression_matrix(seurat_obj)

  expect_true(is.matrix(expr_matrix))
  expect_equal(nrow(expr_matrix), 100)  # cells as rows
  expect_equal(ncol(expr_matrix), 50)   # genes as columns
})

test_that("extract_expression_matrix can transpose", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)

  expr_matrix <- extract_expression_matrix(
    seurat_obj,
    transpose = FALSE
  )

  expect_equal(nrow(expr_matrix), 50)   # genes as rows
  expect_equal(ncol(expr_matrix), 100)  # cells as columns
})

test_that("extract_labels returns factor", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat()

  labels <- extract_labels(seurat_obj, "cell_type")

  expect_true(is.factor(labels))
  expect_equal(length(labels), ncol(seurat_obj))
})

test_that("extract_labels errors on missing column", {
  skip_if_no_seurat()

  seurat_obj <- create_test_seurat()

  expect_error(
    extract_labels(seurat_obj, "nonexistent_column")
  )
})

test_that("create_data_loaders creates list of batches", {
  X <- create_test_matrix(100, 50)
  y <- create_test_labels(100, 3)

  loaders <- create_data_loaders(
    X, y,
    batch_size = 32
  )

  expect_type(loaders, "list")
  expect_true(length(loaders) > 0)

  # Check first batch
  batch <- loaders[[1]]
  expect_true("X" %in% names(batch))
  expect_true("y" %in% names(batch))
  expect_equal(nrow(batch$X), 32)
})

test_that("create_data_loaders handles remainder", {
  X <- create_test_matrix(105, 50)
  y <- create_test_labels(105, 3)

  loaders <- create_data_loaders(
    X, y,
    batch_size = 32
  )

  # Should have 4 batches (32, 32, 32, 9)
  expect_equal(length(loaders), 4)

  # Last batch should have remainder
  expect_equal(nrow(loaders[[4]]$X), 9)
})

test_that("shuffle_data shuffles correctly", {
  set.seed(42)
  X_orig <- create_test_matrix(100, 50)
  y_orig <- create_test_labels(100, 3)

  result <- shuffle_data(X_orig, y_orig, seed = 123)

  # Should have same dimensions
  expect_equal(dim(result$X), dim(X_orig))
  expect_equal(length(result$y), length(y_orig))

  # Should be shuffled (probably)
  expect_false(all(result$y == y_orig))
})

test_that("normalize_data centers and scales", {
  X <- create_test_matrix(100, 50)

  X_norm <- normalize_data(X, method = "standard")

  # Columns should have mean ~0 and sd ~1
  col_means <- colMeans(X_norm)
  col_sds <- apply(X_norm, 2, sd)

  expect_true(all(abs(col_means) < 0.1))
  expect_true(all(abs(col_sds - 1) < 0.1))
})

test_that("normalize_data min-max scaling", {
  X <- create_test_matrix(100, 50)

  X_norm <- normalize_data(X, method = "minmax")

  # Values should be between 0 and 1
  expect_true(all(X_norm >= 0))
  expect_true(all(X_norm <= 1))
})

test_that("create_train_val_test_split creates correct proportions", {
  X <- create_test_matrix(1000, 50)
  y <- create_test_labels(1000, 5)

  splits <- create_train_val_test_split(
    X, y,
    train_ratio = 0.7,
    val_ratio = 0.15,
    test_ratio = 0.15
  )

  expect_equal(nrow(splits$X_train), 700)
  expect_equal(nrow(splits$X_val), 150)
  expect_equal(nrow(splits$X_test), 150)
})

test_that("data utilities handle edge cases", {
  # Single sample
  X <- matrix(rnorm(50), nrow = 1)
  y <- factor(1)

  expect_no_error(normalize_data(X))

  # Empty data should error
  expect_error(normalize_data(matrix(nrow = 0, ncol = 50)))
})
