"""Tests for data loading and datasets."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from anndata import AnnData
from celltype_nn.data.dataset import SingleCellDataset, MultiModalDataset
from celltype_nn.data.loader import split_anndata, create_dataloaders


class TestSingleCellDataset:
    """Test SingleCellDataset class."""

    def test_initialization(self, simple_adata):
        """Test dataset initialization."""
        dataset = SingleCellDataset(simple_adata, label_key='cell_type')

        assert len(dataset) == simple_adata.n_obs
        assert dataset.num_features == simple_adata.n_vars
        assert dataset.num_classes == len(np.unique(simple_adata.obs['cell_type']))

    def test_getitem(self, simple_adata):
        """Test __getitem__ method."""
        dataset = SingleCellDataset(simple_adata, label_key='cell_type')
        sample = dataset[0]

        assert 'features' in sample
        assert 'label' in sample
        assert isinstance(sample['features'], torch.Tensor)
        assert isinstance(sample['label'], torch.Tensor)
        assert sample['features'].shape[0] == simple_adata.n_vars

    def test_label_encoding(self, simple_adata):
        """Test label encoding."""
        dataset = SingleCellDataset(simple_adata, label_key='cell_type')

        assert len(dataset.label_encoder) == dataset.num_classes
        assert len(dataset.idx_to_label) == dataset.num_classes

        # Check bijection
        for label, idx in dataset.label_encoder.items():
            assert dataset.idx_to_label[idx] == label

    def test_get_label_name(self, simple_adata):
        """Test getting label name from index."""
        dataset = SingleCellDataset(simple_adata, label_key='cell_type')

        for idx in range(dataset.num_classes):
            label_name = dataset.get_label_name(idx)
            assert label_name in dataset.label_encoder

    def test_sparse_matrix_conversion(self, simple_adata):
        """Test conversion of sparse matrices."""
        from scipy.sparse import csr_matrix

        # Make data sparse
        simple_adata.X = csr_matrix(simple_adata.X)
        dataset = SingleCellDataset(simple_adata, label_key='cell_type')

        sample = dataset[0]
        assert isinstance(sample['features'], torch.Tensor)
        # Should be converted to dense
        assert not hasattr(sample['features'], 'toarray')

    def test_missing_label_key_error(self, simple_adata):
        """Test error when label key is missing."""
        with pytest.raises(ValueError):
            SingleCellDataset(simple_adata, label_key='nonexistent_key')

    def test_dataloader_compatibility(self, simple_adata):
        """Test compatibility with DataLoader."""
        dataset = SingleCellDataset(simple_adata, label_key='cell_type')
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        batch = next(iter(loader))
        assert batch['features'].shape[0] <= 16
        assert batch['label'].shape[0] <= 16


class TestDataSplitting:
    """Test data splitting functions."""

    def test_split_anndata(self, larger_adata):
        """Test train/val/test splitting."""
        train, val, test = split_anndata(
            larger_adata,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            label_key='cell_type',
            stratify=True,
        )

        assert train.n_obs + val.n_obs + test.n_obs == larger_adata.n_obs
        assert abs(train.n_obs / larger_adata.n_obs - 0.7) < 0.05
        assert abs(val.n_obs / larger_adata.n_obs - 0.15) < 0.05
        assert abs(test.n_obs / larger_adata.n_obs - 0.15) < 0.05

    def test_stratified_splitting(self, larger_adata):
        """Test that stratified splitting preserves class distribution."""
        train, val, test = split_anndata(
            larger_adata,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            label_key='cell_type',
            stratify=True,
        )

        original_dist = larger_adata.obs['cell_type'].value_counts(normalize=True)
        train_dist = train.obs['cell_type'].value_counts(normalize=True)

        # Class distributions should be similar
        for cell_type in original_dist.index:
            if cell_type in train_dist.index:
                assert abs(original_dist[cell_type] - train_dist[cell_type]) < 0.1

    def test_non_stratified_splitting(self, larger_adata):
        """Test non-stratified splitting."""
        train, val, test = split_anndata(
            larger_adata,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            label_key='cell_type',
            stratify=False,
        )

        assert train.n_obs + val.n_obs + test.n_obs == larger_adata.n_obs


class TestCreateDataloaders:
    """Test dataloader creation."""

    def test_create_dataloaders(self, larger_adata):
        """Test creating train/val/test dataloaders."""
        dataloaders = create_dataloaders(
            larger_adata,
            label_key='cell_type',
            batch_size=32,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
        )

        assert 'train' in dataloaders
        assert 'val' in dataloaders
        assert 'test' in dataloaders
        assert 'datasets' in dataloaders

    def test_dataloader_batch_size(self, larger_adata):
        """Test that batch size is respected."""
        dataloaders = create_dataloaders(
            larger_adata,
            label_key='cell_type',
            batch_size=32,
        )

        batch = next(iter(dataloaders['train']))
        assert batch['features'].shape[0] <= 32

    def test_train_shuffle(self, larger_adata):
        """Test that training data is shuffled."""
        dataloaders = create_dataloaders(
            larger_adata,
            label_key='cell_type',
            batch_size=32,
        )

        # Get first batch twice
        iterator1 = iter(dataloaders['train'])
        batch1 = next(iterator1)

        iterator2 = iter(dataloaders['train'])
        batch2 = next(iterator2)

        # Should be different due to shuffling (probabilistically)
        # Note: This test might rarely fail by chance
        assert not torch.equal(batch1['features'], batch2['features'])

    def test_val_no_shuffle(self, larger_adata):
        """Test that validation data is not shuffled."""
        dataloaders = create_dataloaders(
            larger_adata,
            label_key='cell_type',
            batch_size=32,
        )

        # Get first batch twice
        iterator1 = iter(dataloaders['val'])
        batch1 = next(iterator1)

        iterator2 = iter(dataloaders['val'])
        batch2 = next(iterator2)

        # Should be identical (no shuffling)
        assert torch.equal(batch1['features'], batch2['features'])
        assert torch.equal(batch1['label'], batch2['label'])


class TestDataIntegration:
    """Integration tests for data pipeline."""

    def test_end_to_end_data_pipeline(self, larger_adata):
        """Test complete data pipeline."""
        # Create dataloaders
        dataloaders = create_dataloaders(
            larger_adata,
            label_key='cell_type',
            batch_size=64,
            stratify=True,
        )

        # Get datasets
        train_dataset = dataloaders['datasets']['train']
        val_dataset = dataloaders['datasets']['val']
        test_dataset = dataloaders['datasets']['test']

        # Check all have same number of features
        assert train_dataset.num_features == val_dataset.num_features
        assert val_dataset.num_features == test_dataset.num_features

        # Check all have same number of classes
        assert train_dataset.num_classes == val_dataset.num_classes
        assert val_dataset.num_classes == test_dataset.num_classes

        # Check label encoders are consistent
        assert train_dataset.label_encoder == val_dataset.label_encoder
        assert val_dataset.label_encoder == test_dataset.label_encoder

    def test_full_epoch_iteration(self, larger_adata):
        """Test iterating through full epoch."""
        dataloaders = create_dataloaders(
            larger_adata,
            label_key='cell_type',
            batch_size=64,
        )

        total_samples = 0
        for batch in dataloaders['train']:
            total_samples += batch['features'].shape[0]

        train_dataset = dataloaders['datasets']['train']
        assert total_samples == len(train_dataset)
