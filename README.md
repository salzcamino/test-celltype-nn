# CellType-NN: Deep Learning for Cell Type Prediction

A deep learning framework for predicting cell type annotations from single-cell multi-modal data (scRNA-seq, CITE-seq, ATAC-seq).

## Available Implementations

CellType-NN is available in **two implementations**:

1. **Python (PyTorch)** - Main implementation (this README)
   - PyTorch Lightning framework
   - Scanpy for single-cell analysis
   - See sections below for Python usage

2. **R (Keras/TensorFlow)** - R package implementation
   - Seurat for single-cell analysis
   - Keras/TensorFlow for deep learning
   - See [R_README.md](R_README.md) for R usage

Both implementations provide the same core functionality and produce comparable results. Choose based on your preferred language and existing workflow.

## Features

- **Multi-modal Support**: Handle RNA-seq, protein (CITE-seq), and chromatin accessibility (ATAC-seq) data
- **Flexible Architecture**: Multiple model architectures including feedforward networks, attention mechanisms, and VAEs
- **PyTorch Lightning**: Clean, modular training with automatic logging and checkpointing (Python) / Keras callbacks (R)
- **Scalable**: Designed to handle large single-cell datasets with efficient data loading
- **Comprehensive Evaluation**: Built-in metrics, confusion matrices, and visualization tools
- **Configuration-driven**: Easy experimentation with YAML configuration files
- **Cross-platform**: Available in both Python and R

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/celltype-nn.git
cd celltype-nn

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

Core dependencies include:
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- Scanpy >= 1.9.0
- AnnData >= 0.9.0
- Muon >= 0.1.5 (for multi-modal data)
- scikit-learn, pandas, numpy

See `requirements.txt` for full list.

## Quick Start

### 1. Prepare Your Data

Data should be in AnnData (`.h5ad`) format for single-modal or MuData (`.h5mu`) format for multi-modal:

```python
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Ensure cell type labels are in obs
print(adata.obs['cell_type'].value_counts())
```

### 2. Configure Your Experiment

Edit `configs/config.yaml` or create your own configuration file:

```yaml
data:
  file_path: "data/your_data.h5ad"
  label_key: "cell_type"
  batch_size: 256

model:
  type: "rna_classifier"
  hidden_dims: [512, 256, 128]
  dropout_rate: 0.3

training:
  max_epochs: 100
  learning_rate: 0.001
```

### 3. Train the Model

```bash
python scripts/train.py --config configs/config.yaml
```

### 4. Make Predictions

```bash
python scripts/predict.py \
  --checkpoint checkpoints/best_model.ckpt \
  --data data/test_data.h5ad \
  --output predictions/
```

## Project Structure

```
celltype-nn/
├── src/celltype_nn/           # Main package
│   ├── data/                  # Data loading and datasets
│   │   ├── dataset.py         # PyTorch Dataset classes
│   │   └── loader.py          # Data loading utilities
│   ├── models/                # Neural network architectures
│   │   ├── rna_classifier.py  # RNA-only models
│   │   └── multimodal_classifier.py  # Multi-modal models
│   ├── training/              # Training modules
│   │   └── lightning_module.py  # PyTorch Lightning modules
│   ├── evaluation/            # Metrics and visualization
│   │   └── metrics.py         # Evaluation functions
│   ├── preprocessing/         # Data preprocessing
│   │   └── preprocess.py      # Normalization, HVG selection
│   └── utils/                 # Utility functions
├── scripts/                   # Training and inference scripts
│   ├── train.py              # Main training script
│   └── predict.py            # Inference script
├── configs/                   # Configuration files
│   ├── config.yaml           # RNA baseline config
│   └── multimodal_config.yaml  # Multi-modal config
├── notebooks/                 # Jupyter notebooks (examples)
├── tests/                     # Unit tests
└── requirements.txt           # Python dependencies
```

## Model Architectures

### RNA Classifier

Basic feedforward network for RNA-seq data:

```python
from celltype_nn.models.rna_classifier import RNAClassifier

model = RNAClassifier(
    input_dim=2000,      # Number of genes
    num_classes=10,      # Number of cell types
    hidden_dims=[512, 256, 128],
    dropout_rate=0.3
)
```

### RNA Classifier with Attention

Learns to weight genes by importance:

```python
from celltype_nn.models.rna_classifier import RNAClassifierWithAttention

model = RNAClassifierWithAttention(
    input_dim=2000,
    num_classes=10,
    hidden_dims=[512, 256, 128],
    attention_dim=128
)
```

### Multi-Modal Classifier

Integrates multiple data modalities:

```python
from celltype_nn.models.multimodal_classifier import MultiModalClassifier

model = MultiModalClassifier(
    modality_dims={'rna': 2000, 'protein': 50, 'atac': 10000},
    num_classes=10,
    embedding_dim=64,
    fusion_strategy='concat'  # or 'attention'
)
```

## Configuration Options

### Data Configuration

```yaml
data:
  file_path: "path/to/data.h5ad"
  label_key: "cell_type"          # Column in obs with cell types
  batch_size: 256
  num_workers: 4
  train_size: 0.7                 # Train/val/test split
  val_size: 0.15
  test_size: 0.15
  stratify: true                  # Stratified splitting
```

### Preprocessing Configuration

```yaml
preprocessing:
  min_genes: 200                  # Filter cells with < min_genes
  min_cells: 3                    # Filter genes in < min_cells
  n_top_genes: 2000               # Number of highly variable genes
  normalize: true                 # Normalize to target sum
  log_transform: true             # Log1p transformation
  target_sum: 10000               # CPM normalization target
```

### Model Configuration

```yaml
model:
  type: "rna_classifier"          # Model architecture
  hidden_dims: [512, 256, 128]    # Hidden layer sizes
  dropout_rate: 0.3               # Dropout probability
  batch_norm: true                # Use batch normalization
  activation: "relu"              # Activation function
```

### Training Configuration

```yaml
training:
  max_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001
  optimizer: "adamw"              # adam, adamw, sgd
  scheduler: "cosine"             # cosine, step, plateau, null
  gradient_clip_val: 1.0
  early_stopping_patience: 10
```

## Multi-Modal Usage

For multi-modal data (RNA + CITE-seq + ATAC-seq):

```bash
# Use the multi-modal configuration
python scripts/train.py --config configs/multimodal_config.yaml
```

Multi-modal configuration example:

```yaml
data:
  file_path: "data/multimodal.h5mu"
  modalities: ["rna", "protein", "atac"]
  label_key: "cell_type"

preprocessing:
  rna:
    n_top_genes: 2000
    normalize: true
    log_transform: true
  protein:
    normalize: true
    clr_transform: true      # CLR for CITE-seq
  atac:
    tf_idf: true             # TF-IDF for ATAC-seq

model:
  type: "multimodal"
  embedding_dim: 64
  fusion_strategy: "concat"  # or "attention"
```

## Evaluation

The framework provides comprehensive evaluation metrics:

- **Classification metrics**: Accuracy, F1 (macro/micro/weighted), Precision, Recall
- **Per-class metrics**: F1, precision, recall for each cell type
- **Confusion matrix**: Visual and numerical confusion matrices
- **ROC curves**: Multi-class ROC-AUC scores
- **Classification report**: Detailed per-class performance

Metrics are automatically computed during training and saved during prediction.

## Advanced Features

### Handling Missing Modalities

The `MultiModalWithMissingModalities` model can handle cases where some cells have missing modalities:

```python
from celltype_nn.models.multimodal_classifier import MultiModalWithMissingModalities

model = MultiModalWithMissingModalities(
    modality_dims={'rna': 2000, 'protein': 50},
    num_classes=10,
    shared_dim=128
)
```

### Custom Preprocessing

```python
from celltype_nn.preprocessing.preprocess import preprocess_rna

adata = preprocess_rna(
    adata,
    n_top_genes=3000,
    normalize=True,
    log_transform=True,
    scale=False  # Don't scale for neural networks
)
```

### Batch Correction

```python
from celltype_nn.preprocessing.preprocess import batch_correction

adata = batch_correction(
    adata,
    batch_key='batch',
    method='harmony'  # or 'combat'
)
```

## Experiment Tracking

### TensorBoard

Training automatically logs to TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

### Weights & Biases

Enable W&B logging in your config:

```yaml
logging:
  use_wandb: true
  wandb_project: "celltype-nn"
  wandb_entity: "your-entity"
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code
black src/ scripts/

# Sort imports
isort src/ scripts/

# Lint
flake8 src/ scripts/
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{celltype_nn,
  title = {CellType-NN: Deep Learning for Cell Type Prediction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/celltype-nn}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- See example notebooks in `notebooks/`

## R Implementation

If you prefer working in R, we provide a full R implementation with Seurat and Keras. See **[R_README.md](R_README.md)** for:

- Installation instructions for R package
- Quick start guide with Seurat objects
- Example R scripts and vignettes
- Multi-modal analysis in R
- Complete API documentation

The R implementation provides feature parity with Python and integrates seamlessly with the Seurat ecosystem.

## Acknowledgments

Built with:

**Python Implementation:**
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)
- [Scanpy](https://scanpy.readthedocs.io/)
- [Muon](https://muon.readthedocs.io/)

**R Implementation:**
- [Seurat](https://satijalab.org/seurat/)
- [Keras for R](https://keras.rstudio.com/)
- [TensorFlow for R](https://tensorflow.rstudio.com/)
