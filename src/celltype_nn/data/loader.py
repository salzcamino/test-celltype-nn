"""Data loading utilities for creating train/val/test splits."""

from typing import Tuple, Optional, Dict
import numpy as np
from pathlib import Path
import scanpy as sc
from anndata import AnnData
from mudata import MuData
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from .dataset import SingleCellDataset, MultiModalDataset


def load_anndata(file_path: str) -> AnnData:
    """Load AnnData from file.

    Args:
        file_path: Path to h5ad file

    Returns:
        Loaded AnnData object
    """
    return sc.read_h5ad(file_path)


def load_mudata(file_path: str) -> MuData:
    """Load MuData from file.

    Args:
        file_path: Path to h5mu file

    Returns:
        Loaded MuData object
    """
    from mudata import read as read_mudata
    return read_mudata(file_path)


def split_anndata(
    adata: AnnData,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    label_key: str = "cell_type",
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[AnnData, AnnData, AnnData]:
    """Split AnnData into train, validation, and test sets.

    Args:
        adata: AnnData object to split
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        label_key: Key in adata.obs for stratification
        stratify: Whether to stratify by label_key
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_adata, val_adata, test_adata)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "train_size + val_size + test_size must equal 1.0"

    n_obs = adata.n_obs
    indices = np.arange(n_obs)

    stratify_labels = adata.obs[label_key].values if stratify else None

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_size,
        stratify=stratify_labels,
        random_state=random_state
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    temp_labels = adata.obs[label_key].values[temp_indices] if stratify else None

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio,
        stratify=temp_labels,
        random_state=random_state
    )

    return (
        adata[train_indices].copy(),
        adata[val_indices].copy(),
        adata[test_indices].copy()
    )


def create_dataloaders(
    adata: AnnData,
    label_key: str = "cell_type",
    batch_size: int = 256,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    use_raw: bool = False,
    stratify: bool = True,
    num_workers: int = 0,
    random_state: int = 42,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders from AnnData.

    Args:
        adata: AnnData object containing expression data
        label_key: Key in adata.obs containing cell type labels
        batch_size: Batch size for DataLoaders
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        use_raw: Whether to use raw counts
        stratify: Whether to stratify splits by cell type
        num_workers: Number of workers for data loading
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Split data
    train_adata, val_adata, test_adata = split_anndata(
        adata,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        label_key=label_key,
        stratify=stratify,
        random_state=random_state
    )

    # Create datasets
    train_dataset = SingleCellDataset(train_adata, label_key=label_key, use_raw=use_raw)
    val_dataset = SingleCellDataset(val_adata, label_key=label_key, use_raw=use_raw)
    test_dataset = SingleCellDataset(test_adata, label_key=label_key, use_raw=use_raw)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }


def create_multimodal_dataloaders(
    mdata: MuData,
    modalities: list,
    label_key: str = "cell_type",
    batch_size: int = 256,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    use_raw: bool = False,
    stratify: bool = True,
    num_workers: int = 0,
    random_state: int = 42,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test DataLoaders from MuData.

    Args:
        mdata: MuData object containing multi-modal data
        modalities: List of modality names to use
        label_key: Key in mdata.obs containing cell type labels
        batch_size: Batch size for DataLoaders
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        use_raw: Whether to use raw counts
        stratify: Whether to stratify splits by cell type
        num_workers: Number of workers for data loading
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Note: For MuData, we need to split the indices and subset
    # This is a simplified version - in practice you'd want more sophisticated handling
    n_obs = mdata.n_obs
    indices = np.arange(n_obs)

    stratify_labels = mdata.obs[label_key].values if stratify else None

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_size,
        stratify=stratify_labels,
        random_state=random_state
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    temp_labels = mdata.obs[label_key].values[temp_indices] if stratify else None

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_ratio,
        stratify=temp_labels,
        random_state=random_state
    )

    # Create MuData subsets
    train_mdata = mdata[train_indices].copy()
    val_mdata = mdata[val_indices].copy()
    test_mdata = mdata[test_indices].copy()

    # Create datasets
    train_dataset = MultiModalDataset(train_mdata, modalities=modalities, label_key=label_key, use_raw=use_raw)
    val_dataset = MultiModalDataset(val_mdata, modalities=modalities, label_key=label_key, use_raw=use_raw)
    test_dataset = MultiModalDataset(test_mdata, modalities=modalities, label_key=label_key, use_raw=use_raw)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }
