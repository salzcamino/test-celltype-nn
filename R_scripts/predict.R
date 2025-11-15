#!/usr/bin/env Rscript

# Predict Cell Types using Trained Model
#
# This script uses a trained celltypeNN model to predict cell types on new data

library(celltypeNN)
library(Seurat)
library(keras)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  cat("Usage: Rscript predict.R <model_path> <data_path> <output_dir>\n")
  cat("\nArguments:\n")
  cat("  model_path  : Path to trained model (.h5 file)\n")
  cat("  data_path   : Path to input data (.rds or .h5ad)\n")
  cat("  output_dir  : Directory to save predictions\n")
  cat("\nOptional arguments:\n")
  cat("  label_encoder=<path>    : Path to label encoder RDS file\n")
  cat("  cell_type_col=<column>  : Cell type column for evaluation (optional)\n")
  quit(status = 1)
}

# Required arguments
model_path <- args[1]
data_path <- args[2]
output_dir <- args[3]

# Optional arguments
label_encoder_path <- NULL
cell_type_col <- NULL

# Parse optional arguments
if (length(args) > 3) {
  for (i in 4:length(args)) {
    arg_split <- strsplit(args[i], "=")[[1]]
    if (length(arg_split) == 2) {
      key <- arg_split[1]
      value <- arg_split[2]

      if (key == "label_encoder") label_encoder_path <- value
      if (key == "cell_type_col") cell_type_col <- value
    }
  }
}

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("=== Cell Type Prediction ===\n\n")
cat(sprintf("Model: %s\n", model_path))
cat(sprintf("Data: %s\n", data_path))
cat(sprintf("Output: %s\n\n", output_dir))

# Load model
cat("Loading model...\n")
model <- keras::load_model_hdf5(model_path)
cat("Model loaded successfully!\n\n")

# Load label encoder if provided
label_encoder <- NULL
if (!is.null(label_encoder_path)) {
  cat(sprintf("Loading label encoder from %s...\n", label_encoder_path))
  label_encoder <- readRDS(label_encoder_path)
  cat(sprintf("Loaded %d cell type labels\n\n", length(label_encoder)))
}

# Load data
cat("Loading data...\n")
seurat_obj <- load_anndata(data_path)
cat(sprintf("Loaded %d cells and %d genes\n\n", ncol(seurat_obj), nrow(seurat_obj)))

# Preprocess data (use same preprocessing as training)
cat("Preprocessing data...\n")
seurat_obj <- preprocess_rna(
  seurat_obj,
  n_hvgs = 2000,
  min_cells = 3,
  min_features = 200
)

# Create data loader
cat("Preparing data...\n")
if (!is.null(cell_type_col)) {
  data_loader <- create_dataloaders(
    seurat_obj,
    cell_type_col = cell_type_col
  )
} else {
  # No labels available
  expr_matrix <- Seurat::GetAssayData(seurat_obj, slot = "data")
  features <- t(as.matrix(expr_matrix))

  data_loader <- list(
    features = features,
    labels = NULL,
    n_features = ncol(features)
  )
}

# Make predictions
cat("\nMaking predictions...\n")
predictions_probs <- keras::predict(model, data_loader$features)
predicted_classes <- apply(predictions_probs, 1, which.max) - 1

# Decode labels
if (!is.null(label_encoder)) {
  reverse_encoder <- names(label_encoder)
  predicted_labels <- reverse_encoder[predicted_classes + 1]
} else {
  predicted_labels <- paste0("celltype_", predicted_classes)
}

predictions <- list(
  predictions = predicted_labels,
  probabilities = predictions_probs,
  predicted_indices = predicted_classes,
  true_labels = if (!is.null(data_loader$labels)) {
    if (!is.null(label_encoder)) {
      names(label_encoder)[data_loader$labels + 1]
    } else {
      data_loader$labels
    }
  } else {
    NULL
  }
)

cat(sprintf("Predicted %d cells\n", length(predicted_labels)))

# Display prediction summary
cat("\nPrediction summary:\n")
pred_table <- table(predicted_labels)
for (ct in names(pred_table)) {
  cat(sprintf("  %s: %d cells (%.1f%%)\n",
              ct, pred_table[ct],
              100 * pred_table[ct] / length(predicted_labels)))
}

# Save predictions
cat("\nSaving predictions...\n")
save_predictions(
  predictions,
  file.path(output_dir, "predictions.csv"),
  include_probabilities = TRUE
)

# If true labels are available, compute metrics
if (!is.null(predictions$true_labels)) {
  cat("\nEvaluating predictions...\n")
  metrics <- evaluate_model(predictions, compute_per_class = TRUE)

  # Print classification report
  cat("\n")
  cat(classification_report(metrics))

  # Save metrics
  saveRDS(metrics, file.path(output_dir, "metrics.rds"))

  # Save plots
  cm_plot <- plot_confusion_matrix(metrics$confusion_matrix, normalize = TRUE)
  ggplot2::ggsave(
    file.path(output_dir, "confusion_matrix.png"),
    cm_plot,
    width = 10,
    height = 8
  )

  metrics_plot <- plot_metrics(metrics)
  ggplot2::ggsave(
    file.path(output_dir, "per_class_metrics.png"),
    metrics_plot,
    width = 12,
    height = 6
  )

  cat(sprintf("\n✓ Evaluation complete! Results saved to: %s\n", output_dir))
} else {
  cat(sprintf("\n✓ Predictions complete! Results saved to: %s\n", output_dir))
}

cat(sprintf("  - Predictions: %s\n", file.path(output_dir, "predictions.csv")))
if (!is.null(predictions$true_labels)) {
  cat(sprintf("  - Metrics: %s\n", file.path(output_dir, "metrics.rds")))
  cat(sprintf("  - Plots: %s/*.png\n", output_dir))
}
