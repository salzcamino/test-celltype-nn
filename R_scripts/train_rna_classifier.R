#!/usr/bin/env Rscript

# Train RNA-based Cell Type Classifier
#
# This script demonstrates how to train a deep learning model for cell type
# prediction from single-cell RNA-seq data using the celltypeNN package.

library(celltypeNN)
library(Seurat)
library(keras)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript train_rna_classifier.R <input_file> <output_dir>\n")
  cat("\nArguments:\n")
  cat("  input_file  : Path to input Seurat object (.rds) or H5AD file (.h5ad)\n")
  cat("  output_dir  : Directory to save trained model and results\n")
  cat("\nOptional arguments (key=value format):\n")
  cat("  cell_type_col=<column>  : Name of cell type column (default: cell_type)\n")
  cat("  n_hvgs=<number>         : Number of highly variable genes (default: 2000)\n")
  cat("  epochs=<number>         : Number of training epochs (default: 100)\n")
  cat("  batch_size=<number>     : Batch size (default: 64)\n")
  cat("  learning_rate=<number>  : Learning rate (default: 0.001)\n")
  quit(status = 1)
}

# Required arguments
input_file <- args[1]
output_dir <- args[2]

# Optional arguments with defaults
cell_type_col <- "cell_type"
n_hvgs <- 2000
epochs <- 100
batch_size <- 64
learning_rate <- 0.001
hidden_dims <- c(512, 256, 128)
dropout_rate <- 0.3

# Parse optional arguments
if (length(args) > 2) {
  for (i in 3:length(args)) {
    arg_split <- strsplit(args[i], "=")[[1]]
    if (length(arg_split) == 2) {
      key <- arg_split[1]
      value <- arg_split[2]

      if (key == "cell_type_col") cell_type_col <- value
      if (key == "n_hvgs") n_hvgs <- as.numeric(value)
      if (key == "epochs") epochs <- as.numeric(value)
      if (key == "batch_size") batch_size <- as.numeric(value)
      if (key == "learning_rate") learning_rate <- as.numeric(value)
    }
  }
}

# Create output directory
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("=== Cell Type Classifier Training ===\n\n")
cat("Configuration:\n")
cat(sprintf("  Input file: %s\n", input_file))
cat(sprintf("  Output directory: %s\n", output_dir))
cat(sprintf("  Cell type column: %s\n", cell_type_col))
cat(sprintf("  Number of HVGs: %d\n", n_hvgs))
cat(sprintf("  Epochs: %d\n", epochs))
cat(sprintf("  Batch size: %d\n", batch_size))
cat(sprintf("  Learning rate: %.4f\n\n", learning_rate))

# Load data
cat("Loading data...\n")
seurat_obj <- load_anndata(input_file)
cat(sprintf("Loaded %d cells and %d genes\n\n", ncol(seurat_obj), nrow(seurat_obj)))

# Preprocess data
cat("Preprocessing RNA data...\n")
seurat_obj <- preprocess_rna(
  seurat_obj,
  n_hvgs = n_hvgs,
  min_cells = 3,
  min_features = 200
)

# Split data
cat("Splitting data into train/validation/test sets...\n")
data_splits <- split_data(
  seurat_obj,
  train_frac = 0.7,
  val_frac = 0.15,
  test_frac = 0.15,
  cell_type_col = cell_type_col,
  seed = 42
)

cat(sprintf("  Training set: %d cells\n", ncol(data_splits$train)))
cat(sprintf("  Validation set: %d cells\n", ncol(data_splits$validation)))
cat(sprintf("  Test set: %d cells\n\n", ncol(data_splits$test)))

# Create data loaders
cat("Creating data loaders...\n")
train_loader <- create_dataloaders(
  data_splits$train,
  cell_type_col = cell_type_col
)

val_loader <- create_dataloaders(
  data_splits$validation,
  cell_type_col = cell_type_col
)

test_loader <- create_dataloaders(
  data_splits$test,
  cell_type_col = cell_type_col
)

cat(sprintf("  Number of features: %d\n", train_loader$n_features))
cat(sprintf("  Number of classes: %d\n\n", train_loader$n_classes))

# Create model
cat("Creating model...\n")
model <- CellTypeClassifier$new(
  n_features = train_loader$n_features,
  n_classes = train_loader$n_classes,
  hidden_dims = hidden_dims,
  dropout_rate = dropout_rate,
  activation = "relu",
  use_batch_norm = TRUE
)

cat("Model architecture:\n")
model$summary()

# Compile model
cat("\nCompiling model...\n")
model$compile_model(
  optimizer = "adam",
  learning_rate = learning_rate,
  loss = "sparse_categorical_crossentropy"
)

# Train model
cat("\nTraining model...\n")
history <- train_model(
  model = model,
  train_data = train_loader,
  val_data = val_loader,
  epochs = epochs,
  batch_size = batch_size,
  early_stopping_patience = 10,
  checkpoint_dir = file.path(output_dir, "checkpoints"),
  verbose = 1
)

# Plot training history
cat("\nPlotting training history...\n")
history_plot <- plot_training_history(history)
ggplot2::ggsave(
  file.path(output_dir, "training_history.png"),
  history_plot,
  width = 10,
  height = 8
)

# Evaluate on test set
cat("\nEvaluating on test set...\n")
predictions <- predict_celltypes(
  model = model,
  data_loader = test_loader,
  label_encoder = train_loader$label_encoder
)

metrics <- evaluate_model(predictions, compute_per_class = TRUE)

# Print classification report
cat("\n")
cat(classification_report(metrics))

# Save results
cat("\nSaving results...\n")

# Save predictions
save_predictions(
  predictions,
  file.path(output_dir, "test_predictions.csv"),
  include_probabilities = TRUE
)

# Save metrics
saveRDS(metrics, file.path(output_dir, "test_metrics.rds"))

# Save confusion matrix plot
cm_plot <- plot_confusion_matrix(metrics$confusion_matrix, normalize = TRUE)
ggplot2::ggsave(
  file.path(output_dir, "confusion_matrix.png"),
  cm_plot,
  width = 10,
  height = 8
)

# Save per-class metrics plot
metrics_plot <- plot_metrics(metrics)
ggplot2::ggsave(
  file.path(output_dir, "per_class_metrics.png"),
  metrics_plot,
  width = 12,
  height = 6
)

# Save model
model$save(file.path(output_dir, "model_final.h5"))

# Save label encoder
saveRDS(train_loader$label_encoder, file.path(output_dir, "label_encoder.rds"))

cat(sprintf("\n✓ Training complete! Results saved to: %s\n", output_dir))
cat(sprintf("  - Model: %s\n", file.path(output_dir, "model_final.h5")))
cat(sprintf("  - Predictions: %s\n", file.path(output_dir, "test_predictions.csv")))
cat(sprintf("  - Metrics: %s\n", file.path(output_dir, "test_metrics.rds")))
cat(sprintf("  - Plots: %s/*.png\n", output_dir))
