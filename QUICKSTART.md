# Quick Start Guide for Local PC Training

This guide will help you get started training cell type prediction models on your local PC, whether you have a GPU or not.

## Step 1: Check Your System

First, check if your system meets the requirements:

```bash
python scripts/check_system.py
```

This will show you:
- Your RAM and CPU specs
- Whether you have a compatible GPU
- Estimated memory usage for different dataset sizes
- Recommended configuration based on your hardware

## Step 2: Install Dependencies

### Option A: Full Installation (with GPU support)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (for NVIDIA GPUs)
# Visit https://pytorch.org/get-started/locally/ for your specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
pip install -e .
```

### Option B: CPU-Only Installation (no GPU)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
pip install -e .
```

### Option C: Minimal Installation (fastest)

```bash
pip install torch pytorch-lightning scanpy anndata scikit-learn pandas numpy matplotlib seaborn pyyaml tqdm
```

## Step 3: Prepare Your Data

Your data should be in AnnData (`.h5ad`) format with cell type labels:

```python
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Ensure you have cell type labels
# The labels should be in adata.obs (e.g., adata.obs['cell_type'])
print(adata.obs['cell_type'].value_counts())
```

If you don't have data yet, you can download a test dataset:

```python
import scanpy as sc

# Download PBMC 3k dataset (small, good for testing)
adata = sc.datasets.pbmc3k()

# Preprocess
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Add cell type labels (using leiden clustering for demo)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.5)
adata.obs['cell_type'] = adata.obs['leiden']

# Save
adata.write_h5ad("data/pbmc3k.h5ad")
```

## Step 4: Configure Your Training

Choose the appropriate config based on your hardware:

### For CPU Training (No GPU)

Edit `configs/cpu_config.yaml`:

```yaml
data:
  file_path: "data/pbmc3k.h5ad"  # Your data file
  label_key: "cell_type"
  batch_size: 64  # Small batch for CPU

preprocessing:
  n_top_genes: 1000  # Fewer genes = faster training

model:
  hidden_dims: [256, 128]  # Smaller network for CPU

training:
  max_epochs: 30  # Fewer epochs to start

hardware:
  accelerator: "cpu"
  precision: "32"
```

### For GPU Training

Edit `configs/gpu_config.yaml`:

```yaml
data:
  file_path: "data/pbmc3k.h5ad"
  label_key: "cell_type"
  batch_size: 256  # Larger batch for GPU

preprocessing:
  n_top_genes: 2000

model:
  hidden_dims: [512, 256, 128]  # Full network

training:
  max_epochs: 100

hardware:
  accelerator: "gpu"
  precision: "16-mixed"  # Mixed precision for speed
```

## Step 5: Train Your Model

### CPU Training

```bash
python scripts/train.py --config configs/cpu_config.yaml
```

Expected time for 10,000 cells: **10-30 minutes**

### GPU Training

```bash
python scripts/train.py --config configs/gpu_config.yaml
```

Expected time for 10,000 cells: **2-5 minutes**

## Step 6: Monitor Training

Training progress will be displayed in the console:

```
Epoch 1/50: 100%|████████| 40/40 [00:12<00:00,  3.14it/s, loss=2.123, v_num=0, train/acc=0.456]
Validation: 100%|████████| 10/10 [00:02<00:00,  4.87it/s]

Epoch 1 metrics:
  train/loss: 2.123
  train/acc: 0.456
  val/loss: 1.987
  val/acc: 0.521
  val/f1: 0.489
```

You can also view TensorBoard logs:

```bash
tensorboard --logdir outputs/logs
```

## Step 7: Make Predictions

```bash
python scripts/predict.py \
  --checkpoint outputs/logs/rna_cpu_training/version_0/checkpoints/best-model.ckpt \
  --data data/test_data.h5ad \
  --output predictions/
```

## Memory Optimization Tips

### If you run out of RAM:

1. **Reduce batch size**:
   ```yaml
   batch_size: 32  # or even 16
   ```

2. **Use fewer genes**:
   ```yaml
   n_top_genes: 500  # instead of 2000
   ```

3. **Smaller network**:
   ```yaml
   hidden_dims: [128, 64]  # instead of [512, 256, 128]
   ```

4. **Process data in chunks** - subset your dataset:
   ```python
   adata_subset = adata[:10000].copy()  # Use first 10k cells
   ```

### If GPU runs out of memory:

1. **Reduce batch size**:
   ```yaml
   batch_size: 128  # or 64
   ```

2. **Disable mixed precision**:
   ```yaml
   precision: "32"  # instead of "16-mixed"
   ```

3. **Smaller network**:
   ```yaml
   hidden_dims: [256, 128]
   ```

## Alternative: Free Cloud Options

If your local PC is too limited:

### Google Colab (Free GPU)
- Free Tesla T4 GPU (15-16 GB)
- 12-25 GB RAM
- Perfect for most single-cell datasets
- Upload notebook from `notebooks/example_usage.ipynb`
- https://colab.research.google.com/

### Kaggle Kernels (Free GPU)
- Free Tesla P100 GPU (16 GB)
- 30 GB RAM
- 30 hours/week free GPU time
- https://www.kaggle.com/code

### Lambda Cloud / Vast.ai (Paid but cheap)
- ~$0.20-0.50/hour for GPUs
- Good for larger datasets

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` in config
- Reduce `n_top_genes`
- Use smaller `hidden_dims`

### "Killed" / Process terminated
- Not enough RAM
- Reduce dataset size or use fewer genes
- Close other applications

### Very slow training on CPU
- This is normal! CPU is 5-10x slower than GPU
- Reduce `max_epochs` for initial testing
- Use smaller dataset for experimentation
- Consider using free cloud GPU

### Import errors
- Make sure all dependencies are installed
- Run `pip install -r requirements.txt`
- Check `python scripts/check_system.py`

## Performance Benchmarks

Typical training times on different hardware:

| Hardware | Dataset Size | Training Time (50 epochs) |
|----------|-------------|---------------------------|
| CPU (4 cores) | 5,000 cells | 5-10 minutes |
| CPU (4 cores) | 50,000 cells | 30-60 minutes |
| GTX 1650 (4GB) | 5,000 cells | 1-2 minutes |
| GTX 1650 (4GB) | 50,000 cells | 5-10 minutes |
| RTX 3060 (12GB) | 5,000 cells | <1 minute |
| RTX 3060 (12GB) | 100,000 cells | 3-5 minutes |
| RTX 3090 (24GB) | 500,000 cells | 10-15 minutes |

## Next Steps

Once you have a trained model:

1. **Evaluate performance**: Check confusion matrices and F1 scores
2. **Tune hyperparameters**: Adjust learning rate, network size
3. **Try different architectures**: Test attention models or VAEs
4. **Scale to multi-modal**: Add CITE-seq or ATAC-seq data
5. **Deploy for inference**: Use predictions on new datasets

Happy training! 🚀
