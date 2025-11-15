#' Predict Cell Types
#'
#' Make predictions on new data using a trained model
#'
#' @param model Trained CellTypeClassifier or MultiModalClassifier object
#' @param data_loader Data from create_dataloaders
#' @param label_encoder Label encoder from training data
#' @return List with predictions, probabilities, and true labels
#' @export
#' @importFrom keras predict
predict_celltypes <- function(model, data_loader, label_encoder = NULL) {

  if (inherits(model, "CellTypeClassifier") || inherits(model, "MultiModalClassifier")) {
    keras_model <- model$model
  } else {
    keras_model <- model
  }

  # Get predictions
  predictions_probs <- keras::predict(keras_model, data_loader$features)

  # Get predicted class indices
  predicted_classes <- apply(predictions_probs, 1, which.max) - 1  # 0-indexed

  # Decode labels if encoder provided
  if (!is.null(label_encoder)) {
    # Reverse the label encoder
    reverse_encoder <- names(label_encoder)
    predicted_labels <- reverse_encoder[predicted_classes + 1]
  } else {
    predicted_labels <- predicted_classes
  }

  # Get true labels if available
  true_labels <- NULL
  if (!is.null(data_loader$labels)) {
    if (!is.null(label_encoder)) {
      reverse_encoder <- names(label_encoder)
      true_labels <- reverse_encoder[data_loader$labels + 1]
    } else {
      true_labels <- data_loader$labels
    }
  }

  return(list(
    predictions = predicted_labels,
    probabilities = predictions_probs,
    predicted_indices = predicted_classes,
    true_labels = true_labels
  ))
}


#' Evaluate Model Performance
#'
#' Calculate comprehensive evaluation metrics
#'
#' @param predictions Predictions from predict_celltypes
#' @param compute_per_class Compute per-class metrics (default: TRUE)
#' @return List of evaluation metrics
#' @export
#' @importFrom caret confusionMatrix
evaluate_model <- function(predictions, compute_per_class = TRUE) {

  if (is.null(predictions$true_labels)) {
    stop("True labels not available for evaluation")
  }

  pred <- factor(predictions$predictions)
  true <- factor(predictions$true_labels)

  # Ensure same levels
  all_levels <- unique(c(levels(pred), levels(true)))
  pred <- factor(pred, levels = all_levels)
  true <- factor(true, levels = all_levels)

  # Confusion matrix
  cm <- caret::confusionMatrix(pred, true)

  # Extract metrics
  results <- list(
    accuracy = as.numeric(cm$overall["Accuracy"]),
    kappa = as.numeric(cm$overall["Kappa"]),
    confusion_matrix = cm$table,
    overall_stats = cm$overall,
    by_class_stats = cm$byClass
  )

  # Per-class metrics
  if (compute_per_class) {
    classes <- levels(true)
    per_class <- data.frame(
      class = classes,
      precision = numeric(length(classes)),
      recall = numeric(length(classes)),
      f1_score = numeric(length(classes)),
      support = numeric(length(classes))
    )

    for (i in seq_along(classes)) {
      class_name <- classes[i]

      # Get metrics from confusion matrix byClass
      if (length(classes) == 2) {
        # Binary classification
        precision <- cm$byClass["Precision"]
        recall <- cm$byClass["Recall"]
        f1 <- cm$byClass["F1"]
      } else {
        # Multi-class classification
        class_stats <- cm$byClass[i, ]
        precision <- class_stats["Precision"]
        recall <- class_stats["Recall"]
        f1 <- class_stats["F1"]
      }

      per_class$precision[i] <- ifelse(is.na(precision), 0, precision)
      per_class$recall[i] <- ifelse(is.na(recall), 0, recall)
      per_class$f1_score[i] <- ifelse(is.na(f1), 0, f1)
      per_class$support[i] <- sum(true == class_name)
    }

    results$per_class_metrics <- per_class
  }

  # Macro and weighted averages
  if (compute_per_class) {
    per_class <- results$per_class_metrics

    results$macro_f1 <- mean(per_class$f1_score, na.rm = TRUE)
    results$weighted_f1 <- sum(per_class$f1_score * per_class$support, na.rm = TRUE) / sum(per_class$support)

    results$macro_precision <- mean(per_class$precision, na.rm = TRUE)
    results$weighted_precision <- sum(per_class$precision * per_class$support, na.rm = TRUE) / sum(per_class$support)

    results$macro_recall <- mean(per_class$recall, na.rm = TRUE)
    results$weighted_recall <- sum(per_class$recall * per_class$support, na.rm = TRUE) / sum(per_class$support)
  }

  return(results)
}


#' Plot Confusion Matrix
#'
#' Visualize confusion matrix as a heatmap
#'
#' @param confusion_matrix Confusion matrix from evaluate_model
#' @param normalize Normalize values (default: TRUE)
#' @return ggplot object
#' @export
#' @importFrom ggplot2 ggplot aes geom_tile
plot_confusion_matrix <- function(confusion_matrix, normalize = TRUE) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Please install ggplot2 package")
  }

  # Convert to data frame
  cm_df <- as.data.frame.table(confusion_matrix)
  colnames(cm_df) <- c("True", "Predicted", "Count")

  # Normalize if requested
  if (normalize) {
    # Normalize by row (true labels)
    totals <- aggregate(Count ~ True, data = cm_df, FUN = sum)
    cm_df <- merge(cm_df, totals, by = "True", suffixes = c("", "_total"))
    cm_df$Frequency <- cm_df$Count / cm_df$Count_total
    cm_df$Count_total <- NULL

    fill_label <- "Frequency"
    fill_var <- cm_df$Frequency
  } else {
    fill_label <- "Count"
    fill_var <- cm_df$Count
  }

  # Plot
  p <- ggplot2::ggplot(cm_df, ggplot2::aes(x = Predicted, y = True, fill = fill_var)) +
    ggplot2::geom_tile(color = "white") +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.2f", fill_var)),
                       color = "black", size = 3) +
    ggplot2::scale_fill_gradient(low = "white", high = "steelblue", name = fill_label) +
    ggplot2::labs(title = "Confusion Matrix",
                  x = "Predicted Cell Type",
                  y = "True Cell Type") +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

  return(p)
}


#' Plot Evaluation Metrics
#'
#' Visualize per-class performance metrics
#'
#' @param metrics Metrics from evaluate_model
#' @return ggplot object
#' @export
plot_metrics <- function(metrics) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Please install ggplot2 package")
  }

  if (is.null(metrics$per_class_metrics)) {
    stop("Per-class metrics not available")
  }

  # Reshape data for plotting
  metrics_df <- metrics$per_class_metrics
  metrics_long <- tidyr::pivot_longer(
    metrics_df,
    cols = c("precision", "recall", "f1_score"),
    names_to = "metric",
    values_to = "value"
  )

  # Plot
  p <- ggplot2::ggplot(metrics_long, ggplot2::aes(x = class, y = value, fill = metric)) +
    ggplot2::geom_bar(stat = "identity", position = "dodge") +
    ggplot2::labs(title = "Per-Class Performance Metrics",
                  x = "Cell Type",
                  y = "Score",
                  fill = "Metric") +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)) +
    ggplot2::scale_fill_manual(values = c("precision" = "#1f77b4",
                                          "recall" = "#ff7f0e",
                                          "f1_score" = "#2ca02c")) +
    ggplot2::ylim(0, 1)

  return(p)
}


#' Generate Classification Report
#'
#' Create a detailed text report of classification metrics
#'
#' @param metrics Metrics from evaluate_model
#' @return Character string with formatted report
#' @export
classification_report <- function(metrics) {
  report <- paste0(
    "=== Classification Report ===\n\n",
    sprintf("Overall Accuracy: %.4f\n", metrics$accuracy),
    sprintf("Kappa: %.4f\n\n", metrics$kappa),
    sprintf("Macro-averaged F1: %.4f\n", metrics$macro_f1),
    sprintf("Weighted-averaged F1: %.4f\n\n", metrics$weighted_f1),
    sprintf("Macro-averaged Precision: %.4f\n", metrics$macro_precision),
    sprintf("Weighted-averaged Precision: %.4f\n\n", metrics$weighted_precision),
    sprintf("Macro-averaged Recall: %.4f\n", metrics$macro_recall),
    sprintf("Weighted-averaged Recall: %.4f\n\n", metrics$weighted_recall)
  )

  # Add per-class metrics
  if (!is.null(metrics$per_class_metrics)) {
    report <- paste0(report, "Per-Class Metrics:\n")
    report <- paste0(report, sprintf("%-20s %10s %10s %10s %10s\n",
                                     "Class", "Precision", "Recall", "F1-Score", "Support"))
    report <- paste0(report, strrep("-", 70), "\n")

    for (i in 1:nrow(metrics$per_class_metrics)) {
      row <- metrics$per_class_metrics[i, ]
      report <- paste0(report, sprintf("%-20s %10.4f %10.4f %10.4f %10d\n",
                                       row$class,
                                       row$precision,
                                       row$recall,
                                       row$f1_score,
                                       row$support))
    }
  }

  return(report)
}


#' Save Predictions to File
#'
#' Save model predictions and probabilities to CSV
#'
#' @param predictions Predictions from predict_celltypes
#' @param output_path Path to save predictions
#' @param include_probabilities Include probability scores (default: TRUE)
#' @export
save_predictions <- function(predictions, output_path, include_probabilities = TRUE) {

  results_df <- data.frame(
    predicted_celltype = predictions$predictions
  )

  if (!is.null(predictions$true_labels)) {
    results_df$true_celltype <- predictions$true_labels
  }

  if (include_probabilities && !is.null(predictions$probabilities)) {
    prob_df <- as.data.frame(predictions$probabilities)
    colnames(prob_df) <- paste0("prob_class_", seq_len(ncol(prob_df)))
    results_df <- cbind(results_df, prob_df)
  }

  write.csv(results_df, output_path, row.names = FALSE)
  message(paste("Predictions saved to", output_path))
}
