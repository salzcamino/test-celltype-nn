#!/usr/bin/env python
"""Inference script for cell type prediction."""

import os
import sys
from pathlib import Path
import argparse
import yaml
from typing import Dict, Any

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from celltype_nn.data.dataset import SingleCellDataset, MultiModalDataset
from celltype_nn.data.loader import load_anndata, load_mudata
from celltype_nn.preprocessing.preprocess import preprocess_rna, normalize_mudata
from celltype_nn.training.lightning_module import CellTypeClassifierModule, MultiModalClassifierModule
from celltype_nn.evaluation.metrics import (
    evaluate_model,
    plot_confusion_matrix,
    plot_per_class_metrics,
    generate_classification_report
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def predict(
    checkpoint_path: str,
    data_path: str,
    output_dir: str,
    config_path: str = None,
    batch_size: int = 256,
    label_key: str = "cell_type",
    save_plots: bool = True,
):
    """Run prediction on new data.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to data file (h5ad or h5mu)
        output_dir: Directory to save predictions
        config_path: Optional path to config file
        batch_size: Batch size for inference
        label_key: Key for cell type labels (if available for evaluation)
        save_plots: Whether to save evaluation plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    if config_path:
        config = load_config(config_path)
        model_type = config['model']['type']
        if model_type in ['multimodal', 'multimodal_missing']:
            lightning_module = MultiModalClassifierModule.load_from_checkpoint(checkpoint_path)
        else:
            lightning_module = CellTypeClassifierModule.load_from_checkpoint(checkpoint_path)
    else:
        # Try to infer from checkpoint
        try:
            lightning_module = MultiModalClassifierModule.load_from_checkpoint(checkpoint_path)
        except:
            lightning_module = CellTypeClassifierModule.load_from_checkpoint(checkpoint_path)

    model = lightning_module.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # Load data
    print(f"Loading data from {data_path}...")
    data_path_obj = Path(data_path)

    if data_path_obj.suffix == '.h5mu':
        # Multi-modal data
        mdata = load_mudata(data_path)

        if config_path:
            config = load_config(config_path)
            modalities = config['data']['modalities']
            preprocess_config = config.get('preprocessing', {})
            mdata = normalize_mudata(mdata, modality_params=preprocess_config)
        else:
            modalities = list(mdata.mod.keys())

        # Create dataset
        dataset = MultiModalDataset(
            mdata,
            modalities=modalities,
            label_key=label_key if label_key in mdata.obs.columns else None,
            use_raw=False,
        )

        has_labels = label_key in mdata.obs.columns

    else:
        # Single-modal RNA data
        adata = load_anndata(data_path)

        if config_path:
            config = load_config(config_path)
            preprocess_config = config.get('preprocessing', {})
            adata = preprocess_rna(adata, **preprocess_config)

            if preprocess_config.get('n_top_genes'):
                adata = adata[:, adata.var['highly_variable']].copy()

        # Create dataset
        has_labels = label_key in adata.obs.columns
        if has_labels:
            dataset = SingleCellDataset(adata, label_key=label_key, use_raw=False)
        else:
            # Create temporary labels for prediction-only mode
            adata.obs[label_key] = 0
            dataset = SingleCellDataset(adata, label_key=label_key, use_raw=False)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Get label names
    label_names = [dataset.get_label_name(i) for i in range(dataset.num_classes)]

    # Run prediction
    print("Running inference...")
    all_preds = []
    all_probs = []
    all_labels = [] if has_labels else None

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict) and 'features' in batch:
                # Single-modal
                features = batch['features'].to(device)
                if has_labels:
                    labels = batch['label']
                logits = model(features)

            elif isinstance(batch, dict) and 'label' in batch:
                # Multi-modal
                if has_labels:
                    labels = batch.pop('label')
                else:
                    batch.pop('label', None)

                batch_device = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch_device)

                if has_labels:
                    batch['label'] = labels

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            if has_labels:
                all_labels.append(labels.numpy())

    # Concatenate results
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    if has_labels:
        y_true = np.concatenate(all_labels)

    # Convert predictions to cell type names
    predicted_cell_types = [label_names[pred] for pred in y_pred]

    # Save predictions
    predictions_df = pd.DataFrame({
        'predicted_cell_type': predicted_cell_types,
        'prediction_confidence': y_prob.max(axis=1),
    })

    # Add probability for each class
    for i, label in enumerate(label_names):
        predictions_df[f'prob_{label}'] = y_prob[:, i]

    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # If labels are available, evaluate performance
    if has_labels:
        print("\nEvaluating performance...")

        # Compute metrics
        from celltype_nn.evaluation.metrics import compute_metrics

        metrics = compute_metrics(y_true, y_pred, y_prob, label_names)

        print("\nMetrics:")
        for metric, value in metrics.items():
            if not metric.startswith('f1_') or metric in ['f1_macro', 'f1_micro', 'f1_weighted']:
                print(f"  {metric}: {value:.4f}")

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_path = output_dir / 'metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nMetrics saved to {metrics_path}")

        # Generate classification report
        report = generate_classification_report(
            y_true,
            y_pred,
            label_names,
            output_path=output_dir / 'classification_report.txt'
        )
        print("\nClassification Report:")
        print(report)

        # Save plots if requested
        if save_plots:
            print("\nGenerating plots...")

            # Confusion matrix
            plot_confusion_matrix(
                y_true,
                y_pred,
                label_names,
                save_path=output_dir / 'confusion_matrix.png'
            )

            # Per-class metrics
            plot_per_class_metrics(
                y_true,
                y_pred,
                label_names,
                save_path=output_dir / 'per_class_metrics.png'
            )

            print(f"Plots saved to {output_dir}")

    print("\nPrediction complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Predict cell types using trained model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data file (h5ad or h5mu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./predictions',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--label-key',
        type=str,
        default='cell_type',
        help='Key for cell type labels'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    args = parser.parse_args()

    predict(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output,
        config_path=args.config,
        batch_size=args.batch_size,
        label_key=args.label_key,
        save_plots=not args.no_plots,
    )


if __name__ == '__main__':
    main()
