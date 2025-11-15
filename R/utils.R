#' Load Configuration from YAML File
#'
#' Load training configuration from a YAML file
#'
#' @param config_path Path to YAML configuration file
#' @return List containing configuration parameters
#' @export
#' @importFrom yaml read_yaml
load_config <- function(config_path) {
  if (!file.exists(config_path)) {
    stop(paste("Configuration file not found:", config_path))
  }

  config <- yaml::read_yaml(config_path)
  return(config)
}


#' Get Number of Available Threads
#'
#' Get the number of CPU threads available for parallel processing
#'
#' @return Integer number of threads
#' @export
get_num_threads <- function() {
  return(parallel::detectCores())
}


#' Set Random Seeds
#'
#' Set random seeds for reproducibility across R, numpy, and TensorFlow
#'
#' @param seed Random seed value
#' @export
set_seeds <- function(seed = 42) {
  # Set R seed
  set.seed(seed)

  # Set Python/numpy seed if reticulate is available
  if (requireNamespace("reticulate", quietly = TRUE)) {
    tryCatch({
      np <- reticulate::import("numpy")
      np$random$seed(as.integer(seed))
    }, error = function(e) {
      warning("Could not set numpy seed")
    })
  }

  # Set TensorFlow seed
  if (requireNamespace("tensorflow", quietly = TRUE)) {
    tryCatch({
      tensorflow::tf$random$set_seed(as.integer(seed))
    }, error = function(e) {
      warning("Could not set TensorFlow seed")
    })
  }

  message(paste("Random seeds set to", seed))
}


#' Print Model Summary
#'
#' Print a nicely formatted summary of model architecture and parameters
#'
#' @param model CellTypeClassifier or MultiModalClassifier object
#' @export
print_model_summary <- function(model) {
  if (inherits(model, "CellTypeClassifier") || inherits(model, "MultiModalClassifier")) {
    keras_model <- model$model
  } else {
    keras_model <- model
  }

  cat("=== Model Architecture ===\n\n")
  summary(keras_model)

  # Count parameters
  total_params <- sum(sapply(keras_model$trainable_weights, function(w) prod(dim(w))))
  cat(sprintf("\nTotal trainable parameters: %s\n",
              format(total_params, big.mark = ",")))
}


#' Check GPU Availability
#'
#' Check if GPU is available for TensorFlow/Keras
#'
#' @return Logical indicating GPU availability
#' @export
check_gpu <- function() {
  if (!requireNamespace("tensorflow", quietly = TRUE)) {
    message("TensorFlow not installed")
    return(FALSE)
  }

  tryCatch({
    gpus <- tensorflow::tf$config$list_physical_devices('GPU')
    n_gpus <- length(gpus)

    if (n_gpus > 0) {
      cat(sprintf("✓ GPU available: %d device(s) found\n", n_gpus))
      for (i in seq_along(gpus)) {
        cat(sprintf("  GPU %d: %s\n", i - 1, gpus[[i]]$name))
      }
      return(TRUE)
    } else {
      cat("✗ No GPU found. Using CPU.\n")
      return(FALSE)
    }
  }, error = function(e) {
    cat("✗ Error checking GPU:", e$message, "\n")
    return(FALSE)
  })
}


#' Memory Usage Summary
#'
#' Print current R memory usage
#'
#' @export
print_memory_usage <- function() {
  mem <- gc(reset = TRUE)
  cat("=== Memory Usage ===\n")
  cat(sprintf("Used: %.2f MB\n", sum(mem[, 2])))
  cat(sprintf("Max used: %.2f MB\n", sum(mem[, 6])))
}


#' Convert Seurat to AnnData Format (via reticulate)
#'
#' Convert Seurat object to AnnData format for compatibility with Python tools
#'
#' @param seurat_obj Seurat object
#' @param output_path Path to save H5AD file
#' @export
seurat_to_anndata <- function(seurat_obj, output_path) {
  if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
    stop("Please install SeuratDisk: devtools::install_github('mojaveazure/seurat-disk')")
  }

  # Convert to h5Seurat format first
  temp_h5seurat <- tempfile(fileext = ".h5seurat")
  SeuratDisk::SaveH5Seurat(seurat_obj, filename = temp_h5seurat, overwrite = TRUE)

  # Convert to h5ad
  SeuratDisk::Convert(temp_h5seurat, dest = output_path, overwrite = TRUE)

  # Clean up temp file
  unlink(temp_h5seurat)

  message(paste("Seurat object saved as AnnData:", output_path))
}


#' Create Results Directory Structure
#'
#' Create a standardized directory structure for saving results
#'
#' @param output_dir Base output directory
#' @return List of created directory paths
#' @export
create_results_dirs <- function(output_dir) {
  dirs <- list(
    main = output_dir,
    checkpoints = file.path(output_dir, "checkpoints"),
    plots = file.path(output_dir, "plots"),
    predictions = file.path(output_dir, "predictions"),
    metrics = file.path(output_dir, "metrics")
  )

  for (dir_path in dirs) {
    if (!dir.exists(dir_path)) {
      dir.create(dir_path, recursive = TRUE)
    }
  }

  message("Created results directories:")
  for (name in names(dirs)) {
    cat(sprintf("  %s: %s\n", name, dirs[[name]]))
  }

  return(dirs)
}


#' Export Results to CSV
#'
#' Export metrics and predictions to CSV files
#'
#' @param metrics Metrics from evaluate_model
#' @param output_dir Directory to save results
#' @export
export_results <- function(metrics, output_dir) {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Export overall metrics
  overall_metrics <- data.frame(
    metric = c("Accuracy", "Kappa", "Macro_F1", "Weighted_F1",
               "Macro_Precision", "Weighted_Precision",
               "Macro_Recall", "Weighted_Recall"),
    value = c(
      metrics$accuracy,
      metrics$kappa,
      metrics$macro_f1,
      metrics$weighted_f1,
      metrics$macro_precision,
      metrics$weighted_precision,
      metrics$macro_recall,
      metrics$weighted_recall
    )
  )

  write.csv(overall_metrics,
            file.path(output_dir, "overall_metrics.csv"),
            row.names = FALSE)

  # Export per-class metrics
  if (!is.null(metrics$per_class_metrics)) {
    write.csv(metrics$per_class_metrics,
              file.path(output_dir, "per_class_metrics.csv"),
              row.names = FALSE)
  }

  # Export confusion matrix
  cm_df <- as.data.frame.table(metrics$confusion_matrix)
  colnames(cm_df) <- c("True", "Predicted", "Count")
  write.csv(cm_df,
            file.path(output_dir, "confusion_matrix.csv"),
            row.names = FALSE)

  message(paste("Results exported to", output_dir))
}
