# Tests for R6 model classes

test_that("CellTypeClassifier can be initialized", {
  skip_if_no_keras()

  model <- CellTypeClassifier$new(
    n_features = 100,
    n_classes = 5,
    hidden_dims = c(64, 32),
    dropout_rate = 0.3
  )

  expect_equal(model$n_features, 100)
  expect_equal(model$n_classes, 5)
  expect_equal(model$hidden_dims, c(64, 32))
  expect_s3_class(model$model, "keras.engine.training.Model")
})

test_that("CellTypeClassifier model architecture is correct", {
  skip_if_no_keras()

  model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3,
    hidden_dims = c(32, 16)
  )

  # Check model can be summarized
  expect_no_error(model$summary())

  # Model should not be NULL
  expect_false(is.null(model$model))
})

test_that("CellTypeClassifier can compile", {
  skip_if_no_keras()

  model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3
  )

  expect_no_error(
    model$compile_model(
      optimizer = "adam",
      learning_rate = 0.001
    )
  )
})

test_that("CellTypeClassifier handles different activations", {
  skip_if_no_keras()

  for (activation in c("relu", "tanh", "sigmoid")) {
    model <- CellTypeClassifier$new(
      n_features = 50,
      n_classes = 3,
      activation = activation
    )

    expect_false(is.null(model$model))
  }
})

test_that("CellTypeClassifier can be created without batch norm", {
  skip_if_no_keras()

  model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3,
    use_batch_norm = FALSE
  )

  expect_false(is.null(model$model))
})

test_that("MultiModalClassifier can be initialized", {
  skip_if_no_keras()

  model <- MultiModalClassifier$new(
    n_rna_features = 100,
    n_protein_features = 50,
    n_classes = 5,
    embedding_dim = 64
  )

  expect_equal(model$n_rna_features, 100)
  expect_equal(model$n_protein_features, 50)
  expect_equal(model$n_classes, 5)
  expect_s3_class(model$model, "keras.engine.training.Model")
})

test_that("MultiModalClassifier works with RNA only", {
  skip_if_no_keras()

  model <- MultiModalClassifier$new(
    n_rna_features = 100,
    n_protein_features = 0,
    n_atac_features = 0,
    n_classes = 3
  )

  expect_false(is.null(model$model))
})

test_that("MultiModalClassifier works with all modalities", {
  skip_if_no_keras()

  model <- MultiModalClassifier$new(
    n_rna_features = 100,
    n_protein_features = 50,
    n_atac_features = 200,
    n_classes = 5,
    embedding_dim = 32
  )

  expect_false(is.null(model$model))
  expect_no_error(model$summary())
})

test_that("MultiModalClassifier can compile", {
  skip_if_no_keras()

  model <- MultiModalClassifier$new(
    n_rna_features = 100,
    n_protein_features = 50,
    n_classes = 3
  )

  expect_no_error(
    model$compile_model(
      optimizer = "adam",
      learning_rate = 0.001
    )
  )
})

test_that("Model save and load works", {
  skip_if_no_keras()

  model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3
  )

  model$compile_model()

  # Save model
  temp_file <- tempfile(fileext = ".h5")
  expect_no_error(model$save(temp_file))
  expect_true(file.exists(temp_file))

  # Load model
  new_model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3
  )
  expect_no_error(new_model$load(temp_file))

  # Cleanup
  unlink(temp_file)
})

test_that("Model initialization fails with invalid parameters", {
  skip_if_no_keras()

  expect_error(
    CellTypeClassifier$new(
      n_features = -1,
      n_classes = 3
    )
  )

  expect_error(
    CellTypeClassifier$new(
      n_features = 50,
      n_classes = 0
    )
  )
})
