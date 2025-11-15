#' Cell Type Classifier Model
#'
#' R6 Class for RNA-based cell type classification using deep neural networks
#'
#' @description
#' Creates a feedforward neural network for cell type prediction from gene expression data
#'
#' @export
#' @importFrom R6 R6Class
#' @importFrom keras keras_model layer_dense layer_dropout layer_batch_normalization
CellTypeClassifier <- R6::R6Class(
  "CellTypeClassifier",
  public = list(
    #' @field model Keras model object
    model = NULL,

    #' @field n_features Number of input features (genes)
    n_features = NULL,

    #' @field n_classes Number of output classes (cell types)
    n_classes = NULL,

    #' @field hidden_dims Hidden layer dimensions
    hidden_dims = NULL,

    #' @description
    #' Create a new CellTypeClassifier
    #' @param n_features Number of input features
    #' @param n_classes Number of output classes
    #' @param hidden_dims Vector of hidden layer dimensions (default: c(512, 256, 128))
    #' @param dropout_rate Dropout rate (default: 0.3)
    #' @param activation Activation function (default: "relu")
    #' @param use_batch_norm Use batch normalization (default: TRUE)
    initialize = function(n_features,
                          n_classes,
                          hidden_dims = c(512, 256, 128),
                          dropout_rate = 0.3,
                          activation = "relu",
                          use_batch_norm = TRUE) {

      self$n_features <- n_features
      self$n_classes <- n_classes
      self$hidden_dims <- hidden_dims

      # Build model
      self$model <- self$build_model(dropout_rate, activation, use_batch_norm)
    },

    #' @description
    #' Build the neural network architecture
    #' @param dropout_rate Dropout rate
    #' @param activation Activation function
    #' @param use_batch_norm Use batch normalization
    build_model = function(dropout_rate, activation, use_batch_norm) {
      # Input layer
      input_layer <- keras::layer_input(shape = c(self$n_features))
      x <- input_layer

      # Hidden layers
      for (hidden_dim in self$hidden_dims) {
        x <- keras::layer_dense(x, units = hidden_dim)

        if (use_batch_norm) {
          x <- keras::layer_batch_normalization(x)
        }

        x <- keras::layer_activation(x, activation = activation)
        x <- keras::layer_dropout(x, rate = dropout_rate)
      }

      # Output layer
      output_layer <- keras::layer_dense(x, units = self$n_classes, activation = "softmax")

      # Create model
      model <- keras::keras_model(inputs = input_layer, outputs = output_layer)

      return(model)
    },

    #' @description
    #' Compile the model with optimizer and loss function
    #' @param optimizer Optimizer (default: "adam")
    #' @param learning_rate Learning rate (default: 0.001)
    #' @param loss Loss function (default: "sparse_categorical_crossentropy")
    compile_model = function(optimizer = "adam",
                            learning_rate = 0.001,
                            loss = "sparse_categorical_crossentropy") {

      # Create optimizer with learning rate
      if (optimizer == "adam") {
        opt <- keras::optimizer_adam(learning_rate = learning_rate)
      } else if (optimizer == "adamw") {
        # Note: keras may not have adamw, use adam with weight decay
        opt <- keras::optimizer_adam(learning_rate = learning_rate)
      } else if (optimizer == "sgd") {
        opt <- keras::optimizer_sgd(learning_rate = learning_rate, momentum = 0.9)
      } else {
        opt <- optimizer
      }

      self$model %>%
        keras::compile(
          optimizer = opt,
          loss = loss,
          metrics = c("accuracy")
        )
    },

    #' @description
    #' Get model summary
    summary = function() {
      summary(self$model)
    },

    #' @description
    #' Save model to file
    #' @param filepath Path to save the model
    save = function(filepath) {
      keras::save_model_hdf5(self$model, filepath)
      message(paste("Model saved to", filepath))
    },

    #' @description
    #' Load model from file
    #' @param filepath Path to load the model from
    load = function(filepath) {
      self$model <- keras::load_model_hdf5(filepath)
      message(paste("Model loaded from", filepath))
    }
  )
)


#' Multi-Modal Cell Type Classifier
#'
#' R6 Class for multi-modal cell type classification using multiple data types
#'
#' @description
#' Creates a neural network that handles RNA, protein, and ATAC data simultaneously
#'
#' @export
#' @importFrom R6 R6Class
MultiModalClassifier <- R6::R6Class(
  "MultiModalClassifier",
  public = list(
    #' @field model Keras model object
    model = NULL,

    #' @field n_rna_features Number of RNA features
    n_rna_features = NULL,

    #' @field n_protein_features Number of protein features
    n_protein_features = NULL,

    #' @field n_atac_features Number of ATAC features
    n_atac_features = NULL,

    #' @field n_classes Number of output classes
    n_classes = NULL,

    #' @description
    #' Create a new MultiModalClassifier
    #' @param n_rna_features Number of RNA features
    #' @param n_protein_features Number of protein features
    #' @param n_atac_features Number of ATAC features
    #' @param n_classes Number of output classes
    #' @param embedding_dim Embedding dimension for each modality (default: 64)
    #' @param hidden_dims Hidden layer dimensions after fusion (default: c(128, 64))
    #' @param dropout_rate Dropout rate (default: 0.3)
    #' @param fusion_method Fusion method: "concat" or "attention" (default: "concat")
    initialize = function(n_rna_features,
                          n_protein_features = 0,
                          n_atac_features = 0,
                          n_classes,
                          embedding_dim = 64,
                          hidden_dims = c(128, 64),
                          dropout_rate = 0.3,
                          fusion_method = "concat") {

      self$n_rna_features <- n_rna_features
      self$n_protein_features <- n_protein_features
      self$n_atac_features <- n_atac_features
      self$n_classes <- n_classes

      # Build model
      self$model <- self$build_model(embedding_dim, hidden_dims, dropout_rate, fusion_method)
    },

    #' @description
    #' Build the multi-modal neural network
    build_model = function(embedding_dim, hidden_dims, dropout_rate, fusion_method) {
      # RNA encoder
      rna_input <- keras::layer_input(shape = c(self$n_rna_features), name = "rna_input")
      rna_encoded <- rna_input %>%
        keras::layer_dense(units = 256, activation = "relu") %>%
        keras::layer_batch_normalization() %>%
        keras::layer_dropout(rate = dropout_rate) %>%
        keras::layer_dense(units = embedding_dim, activation = "relu", name = "rna_embedding")

      embeddings <- list(rna_encoded)
      inputs <- list(rna_input)

      # Protein encoder (if available)
      if (self$n_protein_features > 0) {
        protein_input <- keras::layer_input(shape = c(self$n_protein_features), name = "protein_input")
        protein_encoded <- protein_input %>%
          keras::layer_dense(units = 128, activation = "relu") %>%
          keras::layer_batch_normalization() %>%
          keras::layer_dropout(rate = dropout_rate) %>%
          keras::layer_dense(units = embedding_dim, activation = "relu", name = "protein_embedding")

        embeddings <- c(embeddings, list(protein_encoded))
        inputs <- c(inputs, list(protein_input))
      }

      # ATAC encoder (if available)
      if (self$n_atac_features > 0) {
        atac_input <- keras::layer_input(shape = c(self$n_atac_features), name = "atac_input")
        atac_encoded <- atac_input %>%
          keras::layer_dense(units = 128, activation = "relu") %>%
          keras::layer_batch_normalization() %>%
          keras::layer_dropout(rate = dropout_rate) %>%
          keras::layer_dense(units = embedding_dim, activation = "relu", name = "atac_embedding")

        embeddings <- c(embeddings, list(atac_encoded))
        inputs <- c(inputs, list(atac_input))
      }

      # Fusion
      if (fusion_method == "concat") {
        if (length(embeddings) > 1) {
          fused <- keras::layer_concatenate(embeddings)
        } else {
          fused <- embeddings[[1]]
        }
      } else {
        # Simple attention fusion (simplified version)
        if (length(embeddings) > 1) {
          fused <- keras::layer_concatenate(embeddings)
        } else {
          fused <- embeddings[[1]]
        }
      }

      # Classification head
      x <- fused
      for (hidden_dim in hidden_dims) {
        x <- x %>%
          keras::layer_dense(units = hidden_dim, activation = "relu") %>%
          keras::layer_batch_normalization() %>%
          keras::layer_dropout(rate = dropout_rate)
      }

      output <- x %>%
        keras::layer_dense(units = self$n_classes, activation = "softmax", name = "output")

      # Create model
      model <- keras::keras_model(inputs = inputs, outputs = output)

      return(model)
    },

    #' @description
    #' Compile the model
    compile_model = function(optimizer = "adam",
                            learning_rate = 0.001,
                            loss = "sparse_categorical_crossentropy") {

      if (optimizer == "adam") {
        opt <- keras::optimizer_adam(learning_rate = learning_rate)
      } else if (optimizer == "sgd") {
        opt <- keras::optimizer_sgd(learning_rate = learning_rate, momentum = 0.9)
      } else {
        opt <- optimizer
      }

      self$model %>%
        keras::compile(
          optimizer = opt,
          loss = loss,
          metrics = c("accuracy")
        )
    },

    #' @description
    #' Get model summary
    summary = function() {
      summary(self$model)
    },

    #' @description
    #' Save model to file
    save = function(filepath) {
      keras::save_model_hdf5(self$model, filepath)
    },

    #' @description
    #' Load model from file
    load = function(filepath) {
      self$model <- keras::load_model_hdf5(filepath)
    }
  )
)
