"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import torch
from anndata import AnnData
import pandas as pd


@pytest.fixture
def simple_adata():
    """Create a simple AnnData object for testing."""
    n_cells = 100
    n_genes = 50

    # Random expression data
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Cell metadata
    obs = pd.DataFrame({
        'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_cells),
        'batch': np.random.choice(['batch1', 'batch2'], n_cells),
    })

    # Gene metadata
    var = pd.DataFrame({
        'gene_name': [f'Gene_{i}' for i in range(n_genes)],
    })

    adata = AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def larger_adata():
    """Create a larger AnnData object for testing."""
    n_cells = 1000
    n_genes = 2000

    # Random expression data (sparse-like)
    X = np.random.negative_binomial(2, 0.5, size=(n_cells, n_genes)).astype(np.float32)

    # Cell metadata with more classes
    cell_types = ['T_cell', 'B_cell', 'NK_cell', 'Monocyte', 'DC']
    obs = pd.DataFrame({
        'cell_type': np.random.choice(cell_types, n_cells),
        'batch': np.random.choice(['batch1', 'batch2', 'batch3'], n_cells),
        'n_genes': (X > 0).sum(axis=1),
    })

    # Gene metadata
    var = pd.DataFrame({
        'gene_name': [f'Gene_{i}' for i in range(n_genes)],
        'n_cells': (X > 0).sum(axis=0),
    })

    adata = AnnData(X=X, obs=obs, var=var)
    return adata


@pytest.fixture
def device():
    """Get the device to use for testing."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    return 42
