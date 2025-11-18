# Integration tests for R package

test_that("Complete RNA workflow with Seurat", {
  skip_if_no_seurat()
  skip_if_no_keras()

  # 1. Create test data
  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)

  # 2. Preprocess
  processed <- preprocess_rna_seurat(
    seurat_obj,
    normalize = TRUE,
    n_variable_features = 30
  )

  # 3. Prepare for training
  data <- prepare_data_for_training(
    processed,
    label_column = "cell_type"
  )

  # 4. Split data
  splits <- split_data(
    data$X, data$y,
    train_ratio = 0.7,
    val_ratio = 0.15,
    test_ratio = 0.15
  )

  # 5. Train model
  model <- train_celltype_classifier(
    splits$X_train, splits$y_train,
    splits$X_val, splits$y_val,
    n_features = ncol(splits$X_train),
    n_classes = length(levels(splits$y_train)),
    hidden_dims = c(32, 16),
    epochs = 2,
    verbose = 0
  )

  # 6. Evaluate
  metrics <- evaluate_model(
    model,
    splits$X_test,
    splits$y_test
  )

  expect_type(metrics, "list")
  expect_true("accuracy" %in% names(metrics))
})

test_that("Complete multi-modal workflow", {
  skip_if_no_seurat()
  skip_if_no_keras()

  # Create multi-modal data
  seurat_obj <- create_test_seurat(n_cells = 100, n_genes = 50)

  # Add protein data
  protein_counts <- matrix(
    rpois(100 * 20, lambda = 10),
    nrow = 20,
    ncol = 100
  )
  rownames(protein_counts) <- paste0("Protein_", 1:20)
  seurat_obj[["ADT"]] <- Seurat::CreateAssayObject(counts = protein_counts)

  # Extract both modalities
  rna_data <- extract_expression_matrix(seurat_obj, assay = "RNA")
  protein_data <- extract_expression_matrix(seurat_obj, assay = "ADT")
  labels <- extract_labels(seurat_obj, "cell_type")

  # Create multi-modal model
  model <- MultiModalClassifier$new(
    n_rna_features = ncol(rna_data),
    n_protein_features = ncol(protein_data),
    n_classes = length(levels(labels)),
    embedding_dim = 32
  )

  model$compile_model()

  expect_s3_class(model$model, "keras.engine.training.Model")
})

test_that("Model persistence workflow", {
  skip_if_no_keras()

  # Train a simple model
  X_train <- create_test_matrix(100, 50)
  y_train <- create_test_labels(100, 3)

  model <- train_celltype_classifier(
    X_train, y_train,
    n_features = 50,
    n_classes = 3,
    epochs = 2,
    verbose = 0
  )

  # Save model
  temp_file <- tempfile(fileext = ".h5")
  model$save(temp_file)

  # Create new model and load weights
  new_model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3
  )
  new_model$load(temp_file)

  # Make predictions with both
  X_test <- create_test_matrix(10, 50)

  pred1 <- predict(model$model, X_test)
  pred2 <- predict(new_model$model, X_test)

  # Predictions should be identical
  expect_equal(pred1, pred2)

  unlink(temp_file)
})

test_that("Cross-validation workflow", {
  skip_if_no_keras()

  X <- create_test_matrix(100, 50)
  y <- create_test_labels(100, 3)

  # Simple 3-fold CV
  folds <- 3
  fold_size <- nrow(X) %/% folds

  accuracies <- numeric(folds)

  for (i in 1:folds) {
    # Create fold indices
    test_idx <- ((i-1)*fold_size + 1):(i*fold_size)
    train_idx <- setdiff(1:nrow(X), test_idx)

    # Split data
    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx]
    X_test <- X[test_idx, , drop = FALSE]
    y_test <- y[test_idx]

    # Train and evaluate
    model <- train_celltype_classifier(
      X_train, y_train,
      n_features = 50,
      n_classes = 3,
      epochs = 1,
      verbose = 0
    )

    metrics <- evaluate_model(model, X_test, y_test)
    accuracies[i] <- metrics$accuracy
  }

  expect_equal(length(accuracies), folds)
  expect_true(all(accuracies >= 0))
  expect_true(all(accuracies <= 1))
})

test_that("Batch prediction workflow", {
  skip_if_no_keras()

  # Train model
  X_train <- create_test_matrix(100, 50)
  y_train <- create_test_labels(100, 3)

  model <- train_celltype_classifier(
    X_train, y_train,
    n_features = 50,
    n_classes = 3,
    epochs = 1,
    verbose = 0
  )

  # Make batch predictions
  X_test <- create_test_matrix(50, 50)

  predictions <- predict(model$model, X_test)

  expect_equal(nrow(predictions), 50)
  expect_equal(ncol(predictions), 3)  # 3 classes

  # Predictions should sum to ~1 (probabilities)
  row_sums <- rowSums(predictions)
  expect_true(all(abs(row_sums - 1) < 0.01))
})

test_that("Different model architectures work", {
  skip_if_no_keras()

  X_train <- create_test_matrix(100, 50)
  y_train <- create_test_labels(100, 3)

  architectures <- list(
    c(32),
    c(64, 32),
    c(128, 64, 32),
    c(256, 128, 64, 32)
  )

  for (hidden_dims in architectures) {
    model <- train_celltype_classifier(
      X_train, y_train,
      n_features = 50,
      n_classes = 3,
      hidden_dims = hidden_dims,
      epochs = 1,
      verbose = 0
    )

    expect_s3_class(model, "CellTypeClassifier")
  }
})

test_that("Error handling in complete workflow", {
  skip_if_no_keras()

  X_train <- create_test_matrix(100, 50)
  y_train <- create_test_labels(100, 3)

  # Wrong feature dimension should error
  X_test_wrong <- create_test_matrix(10, 30)  # Wrong number of features

  model <- train_celltype_classifier(
    X_train, y_train,
    n_features = 50,
    n_classes = 3,
    epochs = 1,
    verbose = 0
  )

  expect_error(
    predict(model$model, X_test_wrong)
  )
})
