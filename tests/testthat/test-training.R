# Tests for training functions

test_that("train_celltype_classifier runs without error", {
  skip_if_no_keras()

  X_train <- create_test_matrix(100, 50)
  y_train <- create_test_labels(100, 3)

  X_val <- create_test_matrix(20, 50)
  y_val <- create_test_labels(20, 3)

  model <- train_celltype_classifier(
    X_train, y_train,
    X_val, y_val,
    n_features = 50,
    n_classes = 3,
    hidden_dims = c(32, 16),
    epochs = 2,  # Quick test
    batch_size = 32,
    verbose = 0
  )

  expect_s3_class(model, "CellTypeClassifier")
})

test_that("train_celltype_classifier returns trained model", {
  skip_if_no_keras()

  X_train <- create_test_matrix(100, 50)
  y_train <- create_test_labels(100, 3)

  model <- train_celltype_classifier(
    X_train, y_train,
    n_features = 50,
    n_classes = 3,
    epochs = 1,
    verbose = 0
  )

  # Model should be compiled and ready
  expect_false(is.null(model$model))
})

test_that("train_with_callbacks uses callbacks correctly", {
  skip_if_no_keras()

  X_train <- create_test_matrix(100, 50)
  y_train <- create_test_labels(100, 3)

  X_val <- create_test_matrix(20, 50)
  y_val <- create_test_labels(20, 3)

  # Create temporary checkpoint directory
  checkpoint_dir <- tempdir()

  history <- train_with_callbacks(
    X_train, y_train,
    X_val, y_val,
    n_features = 50,
    n_classes = 3,
    epochs = 2,
    checkpoint_dir = checkpoint_dir,
    early_stopping = TRUE,
    patience = 5,
    verbose = 0
  )

  expect_s3_class(history, "keras_training_history")
})

test_that("fit_model_with_history returns history", {
  skip_if_no_keras()

  model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3
  )
  model$compile_model()

  X_train <- create_test_matrix(100, 50)
  y_train <- as.integer(create_test_labels(100, 3)) - 1

  history <- fit_model_with_history(
    model$model,
    X_train,
    y_train,
    epochs = 2,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 0
  )

  expect_s3_class(history, "keras_training_history")
  expect_true("loss" %in% names(history$metrics))
})

test_that("create_callbacks returns callback list", {
  skip_if_no_keras()

  callbacks <- create_callbacks(
    checkpoint_dir = tempdir(),
    early_stopping = TRUE,
    patience = 10,
    reduce_lr = TRUE
  )

  expect_type(callbacks, "list")
  expect_true(length(callbacks) > 0)
})

test_that("create_callbacks with tensorboard", {
  skip_if_no_keras()

  log_dir <- tempfile()

  callbacks <- create_callbacks(
    tensorboard = TRUE,
    log_dir = log_dir
  )

  expect_type(callbacks, "list")
})

test_that("training handles class imbalance", {
  skip_if_no_keras()

  # Create imbalanced data
  X_train <- create_test_matrix(100, 50)
  y_train <- factor(c(rep(1, 80), rep(2, 15), rep(3, 5)))

  weights <- calculate_class_weights(y_train)

  expect_equal(length(weights), 3)
  expect_true(all(weights > 0))
  # Minority classes should have higher weights
  expect_true(weights[3] > weights[1])
})

test_that("calculate_class_weights returns correct structure", {
  y <- factor(c(rep("A", 50), rep("B", 30), rep("C", 20)))

  weights <- calculate_class_weights(y)

  expect_type(weights, "double")
  expect_equal(length(weights), 3)
  expect_true(all(weights > 0))
})

test_that("training works with different optimizers", {
  skip_if_no_keras()

  X_train <- create_test_matrix(50, 30)
  y_train <- create_test_labels(50, 3)

  for (optimizer in c("adam", "sgd", "rmsprop")) {
    model <- train_celltype_classifier(
      X_train, y_train,
      n_features = 30,
      n_classes = 3,
      optimizer = optimizer,
      epochs = 1,
      verbose = 0
    )

    expect_s3_class(model, "CellTypeClassifier")
  }
})

test_that("training handles different batch sizes", {
  skip_if_no_keras()

  X_train <- create_test_matrix(100, 30)
  y_train <- create_test_labels(100, 3)

  for (batch_size in c(16, 32, 64)) {
    model <- train_celltype_classifier(
      X_train, y_train,
      n_features = 30,
      n_classes = 3,
      batch_size = batch_size,
      epochs = 1,
      verbose = 0
    )

    expect_s3_class(model, "CellTypeClassifier")
  }
})

test_that("training validation split works", {
  skip_if_no_keras()

  X_train <- create_test_matrix(100, 30)
  y_train <- create_test_labels(100, 3)

  model <- train_celltype_classifier(
    X_train, y_train,
    n_features = 30,
    n_classes = 3,
    validation_split = 0.2,
    epochs = 1,
    verbose = 0
  )

  expect_s3_class(model, "CellTypeClassifier")
})
