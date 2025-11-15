# CellType-NN: R Implementation

R implementation of CellType-NN, a deep learning framework for cell type prediction from single-cell multi-modal data.

## Overview

This R package provides a complete implementation of neural network-based cell type classification for single-cell data, including:

- **Data preprocessing**: Quality control, normalization, and feature selection for scRNA-seq, CITE-seq, and ATAC-seq
- **Neural network models**: Feedforward networks and multi-modal architectures using Keras/TensorFlow
- **Training pipeline**: Automated training with early stopping, learning rate scheduling, and model checkpointing
- **Evaluation**: Comprehensive metrics, visualizations, and classification reports

## Installation

### Prerequisites

1. **R** (>= 4.0.0)
2. **Python** (>= 3.7) with TensorFlow

### Install R Package Dependencies

```r
# Install from CRAN
install.packages(c("Seurat", "keras", "tensorflow", "Matrix", "ggplot2",
                   "dplyr", "tidyr", "purrr", "caret", "reticulate",
                   "yaml", "R6"))

# Install from Bioconductor (optional, for additional features)
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c("SingleCellExperiment", "scater"))

# Install SeuratDisk for H5AD file support
remotes::install_github("mojaveazure/seurat-disk")
```

### Install TensorFlow Backend

```r
library(tensorflow)
install_tensorflow()

library(keras)
install_keras()
```

### Install celltypeNN Package

```r
# Install from source
devtools::install()

# Or load for development
devtools::load_all()
```

## Quick Start

### Example: Training an RNA Classifier

```r
library(celltypeNN)
library(Seurat)

# Load data
seurat_obj <- load_anndata("path/to/data.h5ad")

# Preprocess RNA data
seurat_obj <- preprocess_rna(
  seurat_obj,
  n_hvgs = 2000,
  min_cells = 3,
  min_features = 200
)

# Split data
data_splits <- split_data(
  seurat_obj,
  train_frac = 0.7,
  val_frac = 0.15,
  test_frac = 0.15,
  cell_type_col = "cell_type"
)

# Create data loaders
train_loader <- create_dataloaders(data_splits$train, cell_type_col = "cell_type")
val_loader <- create_dataloaders(data_splits$validation, cell_type_col = "cell_type")
test_loader <- create_dataloaders(data_splits$test, cell_type_col = "cell_type")

# Create and compile model
model <- CellTypeClassifier$new(
  n_features = train_loader$n_features,
  n_classes = train_loader$n_classes,
  hidden_dims = c(512, 256, 128),
  dropout_rate = 0.3
)

model$compile_model(
  optimizer = "adam",
  learning_rate = 0.001
)

# Train model
history <- train_model(
  model = model,
  train_data = train_loader,
  val_data = val_loader,
  epochs = 100,
  batch_size = 64,
  early_stopping_patience = 10
)

# Make predictions
predictions <- predict_celltypes(
  model = model,
  data_loader = test_loader,
  label_encoder = train_loader$label_encoder
)

# Evaluate
metrics <- evaluate_model(predictions)
print(classification_report(metrics))

# Visualize results
plot_confusion_matrix(metrics$confusion_matrix)
plot_metrics(metrics)
```

### Using Command-Line Scripts

#### Training

```bash
Rscript R_scripts/train_rna_classifier.R \
  data/pbmc.h5ad \
  results/rna_classifier \
  cell_type_col=cell_type \
  n_hvgs=2000 \
  epochs=100 \
  batch_size=64
```

#### Prediction

```bash
Rscript R_scripts/predict.R \
  results/rna_classifier/model_final.h5 \
  data/new_data.h5ad \
  results/predictions \
  label_encoder=results/rna_classifier/label_encoder.rds \
  cell_type_col=cell_type
```

## Multi-Modal Classification

The package supports multi-modal data (RNA + protein + ATAC):

```r
# Create multi-modal model
multimodal_model <- MultiModalClassifier$new(
  n_rna_features = 2000,
  n_protein_features = 50,
  n_atac_features = 100,
  n_classes = 10,
  embedding_dim = 64,
  fusion_method = "concat"
)

multimodal_model$compile_model(
  optimizer = "adam",
  learning_rate = 0.001
)

# Prepare multi-modal data
train_data <- list(
  rna = rna_features,
  protein = protein_features,
  atac = atac_features,
  labels = labels
)

# Train
history <- train_multimodal_model(
  model = multimodal_model,
  train_data = train_data,
  val_data = val_data,
  epochs = 100,
  batch_size = 64
)
```

## Data Preprocessing

### RNA-seq Preprocessing

```r
seurat_obj <- preprocess_rna(
  seurat_obj,
  n_hvgs = 2000,              # Number of highly variable genes
  min_cells = 3,              # Min cells expressing a gene
  min_features = 200,         # Min features per cell
  normalization_method = "LogNormalize",
  scale_factor = 10000
)
```

### Protein (CITE-seq) Preprocessing

```r
seurat_obj <- preprocess_protein(
  seurat_obj,
  assay = "ADT",
  normalization_method = "CLR"  # Centered log-ratio normalization
)
```

### ATAC-seq Preprocessing

```r
seurat_obj <- preprocess_atac(
  seurat_obj,
  assay = "ATAC",
  normalization_method = "TFIDF"  # TF-IDF normalization
)
```

### Batch Correction

```r
seurat_obj <- batch_correction(
  seurat_obj,
  batch_col = "batch",
  method = "harmony"  # or "combat"
)
```

## Model Architectures

### RNA Classifier

Feedforward neural network for single-cell RNA-seq:

- Input: Gene expression vector (typically 2000 HVGs)
- Hidden layers: Configurable (default: [512, 256, 128])
- Batch normalization and dropout
- Output: Softmax over cell types

### Multi-Modal Classifier

Handles multiple data modalities:

- Separate encoders for RNA, protein, and ATAC
- Fusion strategies: concatenation or attention
- Shared classification head
- Robust to missing modalities

## Evaluation Metrics

The package provides comprehensive evaluation:

- **Overall metrics**: Accuracy, Kappa
- **Per-class metrics**: Precision, Recall, F1-score
- **Averaged metrics**: Macro and weighted averages
- **Visualizations**: Confusion matrix, per-class performance plots

```r
metrics <- evaluate_model(predictions)

# Text report
cat(classification_report(metrics))

# Plots
plot_confusion_matrix(metrics$confusion_matrix)
plot_metrics(metrics)
```

## Data Formats

### Supported Input Formats

- **H5AD** (AnnData): Requires `SeuratDisk` package
- **RDS** (Seurat): Native R format
- **H5Seurat**: Seurat HDF5 format

### Metadata Requirements

Your data must include a cell type annotation column (default: `cell_type`):

```r
# Check if cell type column exists
colnames(seurat_obj@meta.data)

# Or specify a different column name
create_dataloaders(seurat_obj, cell_type_col = "celltype_annotation")
```

## Model Saving and Loading

```r
# Save model
model$save("my_model.h5")

# Save label encoder
saveRDS(train_loader$label_encoder, "label_encoder.rds")

# Load model
model <- CellTypeClassifier$new(...)
model$load("my_model.h5")

# Or load directly
model_keras <- keras::load_model_hdf5("my_model.h5")
```

## Advanced Usage

### Custom Model Architecture

```r
# Create custom architecture
model <- CellTypeClassifier$new(
  n_features = 2000,
  n_classes = 10,
  hidden_dims = c(1024, 512, 256, 128),  # Deeper network
  dropout_rate = 0.5,                    # Higher dropout
  activation = "relu",
  use_batch_norm = TRUE
)
```

### Custom Training Configuration

```r
history <- train_model(
  model = model,
  train_data = train_loader,
  val_data = val_loader,
  epochs = 200,
  batch_size = 128,
  early_stopping_patience = 20,
  checkpoint_dir = "checkpoints",
  verbose = 1
)
```

### Transfer Learning

```r
# Load pretrained model
pretrained <- keras::load_model_hdf5("pretrained_model.h5")

# Fine-tune on new data
history <- train_model(
  model = pretrained,
  train_data = new_train_loader,
  val_data = new_val_loader,
  epochs = 50,
  batch_size = 64
)
```

## Tips and Best Practices

1. **Data Preprocessing**: Always preprocess your data using the same parameters for training and prediction
2. **Batch Size**: Start with 64, increase if you have more memory
3. **Learning Rate**: 0.001 is a good default, reduce to 0.0001 for fine-tuning
4. **Early Stopping**: Use patience=10 for standard training
5. **HVG Selection**: 2000 genes is standard, increase for complex datasets
6. **Class Imbalance**: Consider using weighted loss or data augmentation for rare cell types

## Comparison with Python Implementation

This R implementation provides feature parity with the Python version:

| Feature | Python | R |
|---------|--------|---|
| RNA Classifier | ✓ | ✓ |
| Multi-modal Classifier | ✓ | ✓ |
| Preprocessing Pipeline | ✓ | ✓ |
| Training & Evaluation | ✓ | ✓ |
| Visualization | ✓ | ✓ |
| H5AD Support | ✓ | ✓ (via SeuratDisk) |
| Batch Correction | ✓ | ✓ |

The R implementation uses Seurat for data handling (instead of Scanpy) and Keras for R (instead of PyTorch), but provides the same functionality and produces comparable results.

## Citation

If you use CellType-NN in your research, please cite:

```
[Your citation here]
```

## License

MIT License - see LICENSE file for details.

## Support

For questions and issues:
- GitHub Issues: [repository URL]
- Documentation: [documentation URL]

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
