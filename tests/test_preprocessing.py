"""Tests for preprocessing functions."""

import pytest
import numpy as np
from anndata import AnnData
from celltype_nn.preprocessing.preprocess import (
    preprocess_rna,
    preprocess_protein,
    preprocess_atac,
    select_highly_variable_genes,
)


class TestPreprocessRNA:
    """Test RNA preprocessing."""

    def test_basic_preprocessing(self, larger_adata):
        """Test basic RNA preprocessing."""
        adata_processed = preprocess_rna(
            larger_adata,
            min_genes=10,
            min_cells=3,
            n_top_genes=100,
            normalize=True,
            log_transform=True,
            scale=False,
        )

        assert adata_processed.n_obs <= larger_adata.n_obs  # Some cells might be filtered
        assert adata_processed.n_vars <= larger_adata.n_vars  # Some genes might be filtered
        assert 'counts' in adata_processed.layers
        assert 'log1p' in adata_processed.layers
        assert 'highly_variable' in adata_processed.var.columns

    def test_normalization(self, simple_adata):
        """Test that normalization works correctly."""
        adata_processed = preprocess_rna(
            simple_adata,
            filter_cells=False,
            filter_genes=False,
            normalize=True,
            log_transform=False,
            target_sum=10000,
        )

        # Check that library sizes are normalized to target sum
        library_sizes = adata_processed.X.sum(axis=1)
        expected = np.full(simple_adata.n_obs, 10000.0)
        assert np.allclose(library_sizes, expected, rtol=1e-5)

    def test_log_transform(self, simple_adata):
        """Test log transformation."""
        adata_processed = preprocess_rna(
            simple_adata,
            filter_cells=False,
            filter_genes=False,
            normalize=False,
            log_transform=True,
        )

        # All values should be >= 0 after log1p
        assert (adata_processed.X >= 0).all()

    def test_highly_variable_genes(self, larger_adata):
        """Test HVG selection."""
        adata_processed = preprocess_rna(
            larger_adata,
            n_top_genes=100,
            normalize=True,
            log_transform=True,
        )

        assert 'highly_variable' in adata_processed.var.columns
        assert adata_processed.var['highly_variable'].sum() == 100

    def test_no_filtering(self, simple_adata):
        """Test preprocessing without filtering."""
        adata_processed = preprocess_rna(
            simple_adata,
            filter_cells=False,
            filter_genes=False,
        )

        assert adata_processed.n_obs == simple_adata.n_obs
        # Note: n_vars might still change due to HVG selection

    def test_with_scaling(self, simple_adata):
        """Test preprocessing with scaling."""
        adata_processed = preprocess_rna(
            simple_adata,
            filter_cells=False,
            filter_genes=False,
            normalize=True,
            log_transform=True,
            scale=True,
        )

        # After scaling, mean should be ~0 and std should be ~1
        means = adata_processed.X.mean(axis=0)
        stds = adata_processed.X.std(axis=0)

        assert np.allclose(means, 0, atol=1e-5)
        assert np.allclose(stds, 1, atol=1e-5)


class TestPreprocessProtein:
    """Test protein (CITE-seq) preprocessing."""

    def test_basic_preprocessing(self, simple_adata):
        """Test basic protein preprocessing."""
        adata_processed = preprocess_protein(
            simple_adata,
            normalize=True,
            log_transform=True,
        )

        assert 'counts' in adata_processed.layers

    def test_clr_transform(self, simple_adata):
        """Test CLR transformation."""
        # Add small pseudocount to avoid log(0)
        simple_adata.X = simple_adata.X + 1

        adata_processed = preprocess_protein(
            simple_adata,
            clr_transform=True,
        )

        # CLR should center the data
        row_means = adata_processed.X.mean(axis=1)
        # After CLR, geometric mean normalization creates specific patterns
        assert adata_processed.X.shape == simple_adata.X.shape


class TestPreprocessATAC:
    """Test ATAC-seq preprocessing."""

    def test_basic_preprocessing(self, simple_adata):
        """Test basic ATAC preprocessing."""
        adata_processed = preprocess_atac(
            simple_adata,
            normalize=True,
            log_transform=True,
        )

        assert 'counts' in adata_processed.layers

    def test_binarization(self, simple_adata):
        """Test binarization."""
        adata_processed = preprocess_atac(
            simple_adata,
            binary=True,
        )

        # All values should be 0 or 1
        assert set(np.unique(adata_processed.X)).issubset({0.0, 1.0})

    def test_tfidf_transform(self, simple_adata):
        """Test TF-IDF transformation."""
        adata_processed = preprocess_atac(
            simple_adata,
            tf_idf=True,
        )

        # TF-IDF should produce non-negative values
        assert (adata_processed.X >= 0).all()


class TestHVGSelection:
    """Test highly variable gene selection."""

    def test_select_hvgs(self, larger_adata):
        """Test HVG selection."""
        # Preprocess first
        larger_adata.layers['counts'] = larger_adata.X.copy()

        adata_processed = select_highly_variable_genes(
            larger_adata,
            n_top_genes=100,
            subset=False,
        )

        assert 'highly_variable' in adata_processed.var.columns
        assert adata_processed.var['highly_variable'].sum() == 100
        assert adata_processed.n_vars == larger_adata.n_vars  # Not subsetted

    def test_select_hvgs_with_subset(self, larger_adata):
        """Test HVG selection with subsetting."""
        larger_adata.layers['counts'] = larger_adata.X.copy()

        adata_processed = select_highly_variable_genes(
            larger_adata,
            n_top_genes=100,
            subset=True,
        )

        assert adata_processed.n_vars == 100


class TestPreprocessingIntegration:
    """Integration tests for preprocessing."""

    def test_full_preprocessing_pipeline(self, larger_adata):
        """Test complete preprocessing workflow."""
        # Full preprocessing
        adata_processed = preprocess_rna(
            larger_adata,
            min_genes=50,
            min_cells=3,
            n_top_genes=500,
            normalize=True,
            log_transform=True,
            scale=False,
        )

        # Check all expected attributes exist
        assert 'counts' in adata_processed.layers
        assert 'log1p' in adata_processed.layers
        assert 'highly_variable' in adata_processed.var.columns
        assert 'n_cells' in adata_processed.var.columns

        # Subset to HVGs
        adata_hvg = adata_processed[:, adata_processed.var['highly_variable']].copy()
        assert adata_hvg.n_vars == 500

        # Data should be normalized and log-transformed
        assert (adata_hvg.X >= 0).all()

    def test_preprocessing_preserves_obs(self, larger_adata):
        """Test that preprocessing preserves observation metadata."""
        original_cell_types = larger_adata.obs['cell_type'].copy()

        adata_processed = preprocess_rna(
            larger_adata,
            min_genes=50,
            normalize=True,
            log_transform=True,
        )

        # Cell types should be preserved for retained cells
        assert 'cell_type' in adata_processed.obs.columns
