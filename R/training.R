#' Train Cell Type Classifier
#'
#' Train a neural network model for cell type prediction
#'
#' @param model CellTypeClassifier or MultiModalClassifier object
#' @param train_data Training data (from create_dataloaders)
#' @param val_data Validation data (from create_dataloaders)
#' @param epochs Number of training epochs (default: 100)
#' @param batch_size Batch size (default: 64)
#' @param early_stopping_patience Early stopping patience (default: 10)
#' @param checkpoint_dir Directory to save model checkpoints (default: "./checkpoints")
#' @param verbose Verbosity level (default: 1)
#' @return Training history object
#' @export
#' @importFrom keras fit compile
train_model <- function(model,
                        train_data,
                        val_data,
                        epochs = 100,
                        batch_size = 64,
                        early_stopping_patience = 10,
                        checkpoint_dir = "./checkpoints",
                        verbose = 1) {

  # Create checkpoint directory if it doesn't exist
  if (!dir.exists(checkpoint_dir)) {
    dir.create(checkpoint_dir, recursive = TRUE)
  }

  # Callbacks
  callbacks <- list(
    # Early stopping
    keras::callback_early_stopping(
      monitor = "val_loss",
      patience = early_stopping_patience,
      restore_best_weights = TRUE,
      verbose = 1
    ),

    # Model checkpoint
    keras::callback_model_checkpoint(
      filepath = file.path(checkpoint_dir, "model_best.h5"),
      monitor = "val_loss",
      save_best_only = TRUE,
      verbose = 1
    ),

    # Reduce learning rate on plateau
    keras::callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = 0.5,
      patience = 5,
      min_lr = 1e-7,
      verbose = 1
    )
  )

  # Train model
  message("Starting training...")

  if (inherits(model, "CellTypeClassifier") || inherits(model, "MultiModalClassifier")) {
    keras_model <- model$model
  } else {
    keras_model <- model
  }

  history <- keras::fit(
    keras_model,
    x = train_data$features,
    y = train_data$labels,
    validation_data = list(val_data$features, val_data$labels),
    epochs = epochs,
    batch_size = batch_size,
    callbacks = callbacks,
    verbose = verbose
  )

  message("Training complete!")

  return(history)
}


#' Train Multi-Modal Model
#'
#' Train a multi-modal neural network with multiple input modalities
#'
#' @param model MultiModalClassifier object
#' @param train_data List containing training data for each modality
#' @param val_data List containing validation data for each modality
#' @param epochs Number of training epochs (default: 100)
#' @param batch_size Batch size (default: 64)
#' @param early_stopping_patience Early stopping patience (default: 10)
#' @param checkpoint_dir Directory to save model checkpoints (default: "./checkpoints")
#' @param verbose Verbosity level (default: 1)
#' @return Training history object
#' @export
train_multimodal_model <- function(model,
                                   train_data,
                                   val_data,
                                   epochs = 100,
                                   batch_size = 64,
                                   early_stopping_patience = 10,
                                   checkpoint_dir = "./checkpoints",
                                   verbose = 1) {

  # Create checkpoint directory
  if (!dir.exists(checkpoint_dir)) {
    dir.create(checkpoint_dir, recursive = TRUE)
  }

  # Prepare input lists
  train_inputs <- list()
  val_inputs <- list()

  if (!is.null(train_data$rna)) {
    train_inputs$rna_input <- train_data$rna
    val_inputs$rna_input <- val_data$rna
  }

  if (!is.null(train_data$protein)) {
    train_inputs$protein_input <- train_data$protein
    val_inputs$protein_input <- val_data$protein
  }

  if (!is.null(train_data$atac)) {
    train_inputs$atac_input <- train_data$atac
    val_inputs$atac_input <- val_data$atac
  }

  # Callbacks
  callbacks <- list(
    keras::callback_early_stopping(
      monitor = "val_loss",
      patience = early_stopping_patience,
      restore_best_weights = TRUE,
      verbose = 1
    ),
    keras::callback_model_checkpoint(
      filepath = file.path(checkpoint_dir, "multimodal_best.h5"),
      monitor = "val_loss",
      save_best_only = TRUE,
      verbose = 1
    ),
    keras::callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = 0.5,
      patience = 5,
      min_lr = 1e-7,
      verbose = 1
    )
  )

  # Train model
  message("Starting multi-modal training...")

  history <- keras::fit(
    model$model,
    x = train_inputs,
    y = train_data$labels,
    validation_data = list(val_inputs, val_data$labels),
    epochs = epochs,
    batch_size = batch_size,
    callbacks = callbacks,
    verbose = verbose
  )

  message("Training complete!")

  return(history)
}


#' Plot Training History
#'
#' Visualize training and validation metrics over epochs
#'
#' @param history Training history object from train_model
#' @return ggplot object
#' @export
#' @importFrom ggplot2 ggplot aes geom_line
plot_training_history <- function(history) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Please install ggplot2 package")
  }

  # Extract metrics
  metrics_df <- data.frame(
    epoch = seq_along(history$metrics$loss),
    train_loss = history$metrics$loss,
    val_loss = history$metrics$val_loss,
    train_acc = history$metrics$accuracy,
    val_acc = history$metrics$val_accuracy
  )

  # Reshape for plotting
  metrics_long <- tidyr::pivot_longer(
    metrics_df,
    cols = -epoch,
    names_to = "metric",
    values_to = "value"
  )

  # Separate loss and accuracy
  loss_df <- metrics_long[grepl("loss", metrics_long$metric), ]
  acc_df <- metrics_long[grepl("acc", metrics_long$metric), ]

  # Plot loss
  p1 <- ggplot2::ggplot(loss_df, ggplot2::aes(x = epoch, y = value, color = metric)) +
    ggplot2::geom_line(size = 1.2) +
    ggplot2::labs(title = "Training and Validation Loss",
                  x = "Epoch",
                  y = "Loss") +
    ggplot2::theme_minimal() +
    ggplot2::scale_color_manual(values = c("train_loss" = "#1f77b4", "val_loss" = "#ff7f0e"))

  # Plot accuracy
  p2 <- ggplot2::ggplot(acc_df, ggplot2::aes(x = epoch, y = value, color = metric)) +
    ggplot2::geom_line(size = 1.2) +
    ggplot2::labs(title = "Training and Validation Accuracy",
                  x = "Epoch",
                  y = "Accuracy") +
    ggplot2::theme_minimal() +
    ggplot2::scale_color_manual(values = c("train_acc" = "#1f77b4", "val_acc" = "#ff7f0e"))

  # Combine plots
  if (requireNamespace("patchwork", quietly = TRUE)) {
    combined <- p1 / p2
    return(combined)
  } else {
    return(list(loss = p1, accuracy = p2))
  }
}
