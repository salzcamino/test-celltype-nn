#!/usr/bin/env python
"""Training script for cell type classification."""

import os
import sys
from pathlib import Path
import argparse
import yaml
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from celltype_nn.data.loader import create_dataloaders, create_multimodal_dataloaders, load_anndata, load_mudata
from celltype_nn.preprocessing.preprocess import preprocess_rna, normalize_mudata
from celltype_nn.models.rna_classifier import RNAClassifier, RNAClassifierWithAttention, RNAVariationalAutoencoder
from celltype_nn.models.multimodal_classifier import MultiModalClassifier, MultiModalWithMissingModalities
from celltype_nn.training.lightning_module import CellTypeClassifierModule, MultiModalClassifierModule


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_data(config: Dict[str, Any]):
    """Load and preprocess data based on configuration."""
    data_config = config['data']
    preprocess_config = config.get('preprocessing', {})

    # Check if single-modal or multi-modal
    if 'modalities' in data_config:
        # Multi-modal data
        print(f"Loading multi-modal data from {data_config['file_path']}...")
        mdata = load_mudata(data_config['file_path'])

        # Preprocess
        print("Preprocessing data...")
        mdata = normalize_mudata(mdata, modality_params=preprocess_config)

        # Create dataloaders
        print("Creating dataloaders...")
        dataloaders = create_multimodal_dataloaders(
            mdata,
            modalities=data_config['modalities'],
            label_key=data_config['label_key'],
            batch_size=data_config['batch_size'],
            train_size=data_config['train_size'],
            val_size=data_config['val_size'],
            test_size=data_config['test_size'],
            use_raw=data_config.get('use_raw', False),
            stratify=data_config.get('stratify', True),
            num_workers=data_config.get('num_workers', 0),
        )

        # Get dimensions
        train_dataset = dataloaders['datasets']['train']
        num_features = train_dataset.num_features
        num_classes = train_dataset.num_classes

        return dataloaders, num_features, num_classes, train_dataset.label_encoder

    else:
        # Single-modal RNA data
        print(f"Loading data from {data_config['file_path']}...")
        adata = load_anndata(data_config['file_path'])

        # Preprocess
        print("Preprocessing data...")
        adata = preprocess_rna(adata, **preprocess_config)

        # Subset to highly variable genes if requested
        if preprocess_config.get('n_top_genes'):
            adata = adata[:, adata.var['highly_variable']].copy()

        # Create dataloaders
        print("Creating dataloaders...")
        dataloaders = create_dataloaders(
            adata,
            label_key=data_config['label_key'],
            batch_size=data_config['batch_size'],
            train_size=data_config['train_size'],
            val_size=data_config['val_size'],
            test_size=data_config['test_size'],
            use_raw=data_config.get('use_raw', False),
            stratify=data_config.get('stratify', True),
            num_workers=data_config.get('num_workers', 0),
        )

        # Get dimensions
        train_dataset = dataloaders['datasets']['train']
        num_features = train_dataset.num_features
        num_classes = train_dataset.num_classes

        return dataloaders, num_features, num_classes, train_dataset.label_encoder


def create_model(config: Dict[str, Any], num_features, num_classes):
    """Create model based on configuration."""
    model_config = config['model']
    model_type = model_config['type']

    if model_type == 'rna_classifier':
        model = RNAClassifier(
            input_dim=num_features,
            num_classes=num_classes,
            hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
            dropout_rate=model_config.get('dropout_rate', 0.3),
            batch_norm=model_config.get('batch_norm', True),
            activation=model_config.get('activation', 'relu'),
        )

    elif model_type == 'rna_attention':
        model = RNAClassifierWithAttention(
            input_dim=num_features,
            num_classes=num_classes,
            hidden_dims=model_config.get('hidden_dims', [512, 256, 128]),
            attention_dim=model_config.get('attention_dim', 128),
            dropout_rate=model_config.get('dropout_rate', 0.3),
            batch_norm=model_config.get('batch_norm', True),
        )

    elif model_type == 'rna_vae':
        model = RNAVariationalAutoencoder(
            input_dim=num_features,
            num_classes=num_classes,
            latent_dim=model_config.get('latent_dim', 32),
            hidden_dims=model_config.get('hidden_dims', [512, 256]),
            dropout_rate=model_config.get('dropout_rate', 0.3),
        )

    elif model_type == 'multimodal':
        # num_features is dict for multimodal
        model = MultiModalClassifier(
            modality_dims=num_features,
            num_classes=num_classes,
            embedding_dim=model_config.get('embedding_dim', 64),
            hidden_dims=model_config.get('modality_hidden_dims', [256, 128]),
            classifier_dims=model_config.get('classifier_dims', [128, 64]),
            dropout_rate=model_config.get('dropout_rate', 0.3),
            fusion_strategy=model_config.get('fusion_strategy', 'concat'),
        )

    elif model_type == 'multimodal_missing':
        model = MultiModalWithMissingModalities(
            modality_dims=num_features,
            num_classes=num_classes,
            shared_dim=model_config.get('shared_dim', 128),
            modality_specific_dims=model_config.get('modality_hidden_dims', [256, 128]),
            dropout_rate=model_config.get('dropout_rate', 0.3),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train(config_path: str):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)

    # Set random seed
    pl.seed_everything(config['experiment']['seed'])

    # Setup data
    dataloaders, num_features, num_classes, label_encoder = setup_data(config)

    print(f"\nDataset info:")
    print(f"  Number of features: {num_features}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {list(label_encoder.keys())}")

    # Create model
    print("\nCreating model...")
    model = create_model(config, num_features, num_classes)

    # Create Lightning module
    training_config = config['training']
    model_type = config['model']['type']

    if model_type in ['multimodal', 'multimodal_missing']:
        lightning_module = MultiModalClassifierModule(
            model=model,
            num_classes=num_classes,
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            optimizer=training_config.get('optimizer', 'adamw'),
            scheduler=training_config.get('scheduler', 'cosine'),
        )
    else:
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=num_classes,
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            optimizer=training_config.get('optimizer', 'adamw'),
            scheduler=training_config.get('scheduler', 'cosine'),
            use_focal_loss=training_config.get('use_focal_loss', False),
        )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    if config['logging'].get('save_checkpoints', True):
        checkpoint_dir = Path(config['experiment']['output_dir']) / config['logging']['checkpoint_dir']
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{config['experiment']['name']}-{{epoch:02d}}-{{val/loss:.4f}}",
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Early stopping callback
    if training_config.get('early_stopping_patience'):
        early_stop_callback = EarlyStopping(
            monitor=training_config.get('early_stopping_metric', 'val/loss'),
            patience=training_config['early_stopping_patience'],
            mode=training_config.get('early_stopping_mode', 'min'),
        )
        callbacks.append(early_stop_callback)

    # Setup logger
    loggers = []
    log_dir = Path(config['experiment']['output_dir']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=config['experiment']['name'],
    )
    loggers.append(tb_logger)

    if config['logging'].get('use_wandb', False):
        wandb_logger = WandbLogger(
            project=config['logging']['wandb_project'],
            entity=config['logging'].get('wandb_entity'),
            name=config['experiment']['name'],
            config=config,
        )
        loggers.append(wandb_logger)

    # Create trainer
    hardware_config = config['hardware']
    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        accelerator=hardware_config.get('accelerator', 'auto'),
        devices=hardware_config.get('devices', 1),
        precision=hardware_config.get('precision', '32'),
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        log_every_n_steps=config['logging'].get('log_every_n_steps', 50),
    )

    # Train
    print("\nStarting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val'],
    )

    # Test
    print("\nRunning test evaluation...")
    trainer.test(lightning_module, dataloaders=dataloaders['test'])

    print(f"\nTraining complete! Logs saved to {log_dir}")
    if config['logging'].get('save_checkpoints', True):
        print(f"Checkpoints saved to {checkpoint_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train cell type classifier")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    train(args.config)


if __name__ == '__main__':
    main()
