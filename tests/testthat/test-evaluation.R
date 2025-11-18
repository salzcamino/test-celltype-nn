# Tests for evaluation functions

test_that("evaluate_model returns metrics", {
  skip_if_no_keras()

  model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3
  )
  model$compile_model()

  X_test <- create_test_matrix(50, 50)
  y_test <- create_test_labels(50, 3)

  metrics <- evaluate_model(
    model,
    X_test,
    y_test
  )

  expect_type(metrics, "list")
  expect_true("accuracy" %in% names(metrics))
  expect_true("predictions" %in% names(metrics))
})

test_that("calculate_metrics computes accuracy", {
  y_true <- factor(c(1, 1, 2, 2, 3, 3))
  y_pred <- factor(c(1, 1, 2, 2, 3, 3))  # Perfect predictions

  metrics <- calculate_metrics(y_true, y_pred)

  expect_equal(metrics$accuracy, 1.0)
})

test_that("calculate_metrics computes confusion matrix", {
  y_true <- factor(c(1, 1, 2, 2, 3, 3))
  y_pred <- factor(c(1, 2, 2, 3, 3, 1))  # Some errors

  metrics <- calculate_metrics(y_true, y_pred)

  expect_true("confusion_matrix" %in% names(metrics))
  expect_true(is.matrix(metrics$confusion_matrix))
})

test_that("calculate_metrics computes per-class metrics", {
  y_true <- factor(c(1, 1, 2, 2, 3, 3))
  y_pred <- factor(c(1, 1, 2, 2, 3, 3))

  metrics <- calculate_metrics(y_true, y_pred)

  expect_true("precision" %in% names(metrics))
  expect_true("recall" %in% names(metrics))
  expect_true("f1_score" %in% names(metrics))
})

test_that("plot_confusion_matrix creates plot", {
  y_true <- factor(c(1, 1, 2, 2, 3, 3))
  y_pred <- factor(c(1, 2, 2, 3, 3, 1))

  # Should not error
  expect_no_error(
    plot_confusion_matrix(y_true, y_pred)
  )
})

test_that("plot_confusion_matrix can save to file", {
  y_true <- factor(c(1, 1, 2, 2, 3, 3))
  y_pred <- factor(c(1, 1, 2, 2, 3, 3))

  temp_file <- tempfile(fileext = ".png")

  plot_confusion_matrix(
    y_true, y_pred,
    save_path = temp_file
  )

  expect_true(file.exists(temp_file))
  unlink(temp_file)
})

test_that("calculate_accuracy works correctly", {
  y_true <- c(1, 2, 3, 1, 2, 3)
  y_pred <- c(1, 2, 3, 1, 2, 3)  # Perfect

  acc <- calculate_accuracy(y_true, y_pred)
  expect_equal(acc, 1.0)

  y_pred <- c(1, 1, 1, 1, 1, 1)  # All wrong except first
  acc <- calculate_accuracy(y_true, y_pred)
  expect_equal(acc, 1/6)
})

test_that("calculate_f1_score works correctly", {
  # Binary case
  y_true <- factor(c(1, 1, 1, 0, 0, 0))
  y_pred <- factor(c(1, 1, 1, 0, 0, 0))  # Perfect

  f1 <- calculate_f1_score(y_true, y_pred)
  expect_equal(f1, 1.0)
})

test_that("generate_classification_report creates report", {
  y_true <- factor(c("A", "A", "B", "B", "C", "C"))
  y_pred <- factor(c("A", "A", "B", "B", "C", "C"))

  report <- generate_classification_report(y_true, y_pred)

  expect_type(report, "character")
  expect_true(nchar(report) > 0)
})

test_that("predict_and_evaluate returns predictions and metrics", {
  skip_if_no_keras()

  model <- CellTypeClassifier$new(
    n_features = 50,
    n_classes = 3
  )
  model$compile_model()

  X_test <- create_test_matrix(30, 50)
  y_test <- create_test_labels(30, 3)

  result <- predict_and_evaluate(
    model,
    X_test,
    y_test
  )

  expect_true("predictions" %in% names(result))
  expect_true("metrics" %in% names(result))
  expect_equal(length(result$predictions), 30)
})

test_that("evaluation handles edge cases", {
  # All same prediction
  y_true <- factor(c(1, 2, 3, 1, 2, 3))
  y_pred <- factor(c(1, 1, 1, 1, 1, 1))

  metrics <- calculate_metrics(y_true, y_pred)

  expect_type(metrics, "list")
  expect_true(metrics$accuracy < 1.0)
})

test_that("metrics calculation with class names", {
  y_true <- factor(c("TypeA", "TypeA", "TypeB", "TypeB"))
  y_pred <- factor(c("TypeA", "TypeB", "TypeB", "TypeA"))

  metrics <- calculate_metrics(y_true, y_pred)

  expect_equal(metrics$accuracy, 0.5)
})
