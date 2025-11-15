#!/usr/bin/env python
"""Check system requirements and provide training recommendations."""

import sys
import platform
import psutil
import numpy as np


def format_bytes(bytes):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def check_gpu():
    """Check for GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            return True, gpu_count, gpu_name, gpu_memory
        else:
            return False, 0, None, 0
    except ImportError:
        return False, 0, None, 0


def estimate_memory_usage(n_cells, n_genes, batch_size=256):
    """Estimate memory usage for training.

    Args:
        n_cells: Number of cells in dataset
        n_genes: Number of genes (features)
        batch_size: Batch size for training

    Returns:
        Estimated memory in bytes
    """
    # Data memory: cells × genes × 4 bytes (float32)
    data_memory = n_cells * n_genes * 4

    # Model parameters (rough estimate for 512->256->128 network)
    hidden_dims = [512, 256, 128]
    param_count = n_genes * hidden_dims[0]
    for i in range(len(hidden_dims) - 1):
        param_count += hidden_dims[i] * hidden_dims[i + 1]
    # × 4 bytes per param, × 3 for gradients and optimizer state
    model_memory = param_count * 4 * 3

    # Batch memory during forward/backward pass
    batch_memory = batch_size * n_genes * 4 * 10  # rough multiplier for activations

    # Total estimate with 1.5x safety margin
    total = (data_memory + model_memory + batch_memory) * 1.5

    return total


def get_recommendations(ram_gb, has_gpu, gpu_memory_gb=0):
    """Get training recommendations based on system specs."""
    recommendations = []

    if ram_gb < 8:
        recommendations.append("⚠️  WARNING: Less than 8 GB RAM detected. Training may be challenging.")
        recommendations.append("   - Use very small datasets (<5,000 cells)")
        recommendations.append("   - Reduce batch_size to 32 or 64")
        recommendations.append("   - Reduce n_top_genes to 500-1000")
    elif ram_gb < 16:
        recommendations.append("✓ 8-16 GB RAM: Good for small to medium datasets")
        recommendations.append("   - Recommended: <50,000 cells")
        recommendations.append("   - Use batch_size: 64-128")
        recommendations.append("   - Use n_top_genes: 1000-2000")
    else:
        recommendations.append("✓ 16+ GB RAM: Good for large datasets")
        recommendations.append("   - Can handle 100,000+ cells (with GPU)")
        recommendations.append("   - Use batch_size: 128-256")
        recommendations.append("   - Use n_top_genes: 2000-3000")

    if has_gpu:
        if gpu_memory_gb < 4:
            recommendations.append("⚠️  GPU Memory < 4 GB: Limited GPU training")
            recommendations.append("   - Small batch sizes (32-64)")
            recommendations.append("   - Smaller networks recommended")
        elif gpu_memory_gb < 8:
            recommendations.append("✓ GPU with 4-8 GB: Good for most tasks")
            recommendations.append("   - Use batch_size: 128-256")
            recommendations.append("   - Can train full-size networks")
            recommendations.append("   - Use mixed precision (precision: '16-mixed')")
        else:
            recommendations.append("✓ GPU with 8+ GB: Excellent for deep learning")
            recommendations.append("   - Use batch_size: 256-512")
            recommendations.append("   - Can train large networks")
            recommendations.append("   - Can handle multi-modal models")
    else:
        recommendations.append("ℹ️  No GPU detected: CPU training only")
        recommendations.append("   - Training will be slower (5-10x)")
        recommendations.append("   - Use smaller networks (hidden_dims: [256, 128])")
        recommendations.append("   - Reduce batch_size to 64")
        recommendations.append("   - Consider using Google Colab (free GPU) for larger datasets")

    return recommendations


def main():
    """Main function to check system and provide recommendations."""
    print("=" * 70)
    print("CellType-NN System Requirements Check")
    print("=" * 70)
    print()

    # System info
    print("System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  CPU: {platform.processor() or 'Unknown'}")
    print(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print()

    # Memory info
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print("Memory:")
    print(f"  Total RAM: {format_bytes(ram.total)} ({ram_gb:.1f} GB)")
    print(f"  Available RAM: {format_bytes(ram.available)} ({ram.available / (1024**3):.1f} GB)")
    print()

    # GPU info
    has_gpu, gpu_count, gpu_name, gpu_memory = check_gpu()
    gpu_memory_gb = gpu_memory / (1024**3) if gpu_memory else 0

    print("GPU:")
    if has_gpu:
        print(f"  ✓ GPU Available: Yes")
        print(f"  Count: {gpu_count}")
        print(f"  Name: {gpu_name}")
        print(f"  Memory: {format_bytes(gpu_memory)} ({gpu_memory_gb:.1f} GB)")

        try:
            import torch
            print(f"  PyTorch CUDA: {torch.version.cuda}")
            print(f"  cuDNN: {torch.backends.cudnn.version()}")
        except:
            pass
    else:
        print(f"  ✗ GPU Available: No")
        print(f"  Note: You can still train on CPU, just slower")
    print()

    # Memory usage estimates
    print("Estimated Memory Usage for Different Dataset Sizes:")
    print("-" * 70)
    for n_cells in [1000, 5000, 10000, 50000, 100000]:
        mem = estimate_memory_usage(n_cells, n_genes=2000, batch_size=128)
        mem_gb = mem / (1024**3)
        status = "✓" if mem < ram.available else "⚠️ "
        print(f"  {status} {n_cells:6,} cells × 2000 genes: ~{format_bytes(mem):>10} ({mem_gb:.1f} GB)")
    print()

    # Recommendations
    print("Recommendations:")
    print("-" * 70)
    recommendations = get_recommendations(ram_gb, has_gpu, gpu_memory_gb)
    for rec in recommendations:
        print(rec)
    print()

    # Configuration suggestion
    print("Suggested Configuration File:")
    print("-" * 70)
    if has_gpu:
        print("  Use: configs/gpu_config.yaml")
        print("  Command: python scripts/train.py --config configs/gpu_config.yaml")
    else:
        print("  Use: configs/cpu_config.yaml")
        print("  Command: python scripts/train.py --config configs/cpu_config.yaml")
    print()

    # Installation check
    print("Checking Dependencies:")
    print("-" * 70)
    dependencies = [
        ('torch', 'PyTorch'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('scanpy', 'Scanpy'),
        ('anndata', 'AnnData'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
    ]

    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(name)

    print()
    if missing:
        print("⚠️  Missing dependencies. Install with:")
        print("    pip install -r requirements.txt")
    else:
        print("✓ All dependencies installed!")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
