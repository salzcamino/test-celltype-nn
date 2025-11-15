"""Preprocessing utilities for single-cell data."""

from typing import Optional, Union, List
import numpy as np
import scanpy as sc
from anndata import AnnData
from mudata import MuData


def preprocess_rna(
    adata: AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    n_top_genes: int = 2000,
    normalize: bool = True,
    log_transform: bool = True,
    scale: bool = False,
    target_sum: float = 1e4,
    filter_cells: bool = True,
    filter_genes: bool = True,
) -> AnnData:
    """Preprocess RNA-seq data with standard scanpy workflow.

    Args:
        adata: AnnData object with raw counts
        min_genes: Minimum number of genes per cell for filtering
        min_cells: Minimum number of cells per gene for filtering
        n_top_genes: Number of highly variable genes to select
        normalize: Whether to normalize to target sum
        log_transform: Whether to apply log1p transformation
        scale: Whether to scale to unit variance
        target_sum: Target sum for normalization (typically 1e4 for CPM)
        filter_cells: Whether to filter cells by min_genes
        filter_genes: Whether to filter genes by min_cells

    Returns:
        Preprocessed AnnData object
    """
    adata = adata.copy()

    # Basic QC filtering
    if filter_cells:
        sc.pp.filter_cells(adata, min_genes=min_genes)

    if filter_genes:
        sc.pp.filter_genes(adata, min_cells=min_cells)

    # Store raw counts
    adata.layers['counts'] = adata.X.copy()

    # Normalization
    if normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum)

    # Log transformation
    if log_transform:
        sc.pp.log1p(adata)

    # Store normalized/log-transformed data
    adata.layers['log1p'] = adata.X.copy()

    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', layer='counts')

    # Optional scaling (typically used for visualization, not for NN input)
    if scale:
        sc.pp.scale(adata)

    return adata


def preprocess_protein(
    adata: AnnData,
    normalize: bool = True,
    log_transform: bool = True,
    scale: bool = False,
    clr_transform: bool = False,
) -> AnnData:
    """Preprocess protein (CITE-seq) data.

    Args:
        adata: AnnData object with protein counts
        normalize: Whether to normalize
        log_transform: Whether to apply log1p transformation
        scale: Whether to scale to unit variance
        clr_transform: Whether to apply centered log-ratio transformation

    Returns:
        Preprocessed AnnData object
    """
    adata = adata.copy()

    # Store raw counts
    adata.layers['counts'] = adata.X.copy()

    if clr_transform:
        # CLR transformation (common for CITE-seq)
        sc.pp.normalize_total(adata, target_sum=1e4)
        # Compute CLR: log(x / geometric_mean(x))
        from scipy.stats import gmean
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        X = X + 1  # Pseudocount
        geometric_means = gmean(X, axis=1, keepdims=True)
        adata.X = np.log(X / geometric_means)
    else:
        # Standard normalization + log
        if normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)

        if log_transform:
            adata.X = np.log1p(adata.X)

    if scale:
        sc.pp.scale(adata)

    return adata


def preprocess_atac(
    adata: AnnData,
    binary: bool = False,
    normalize: bool = True,
    log_transform: bool = True,
    tf_idf: bool = False,
) -> AnnData:
    """Preprocess ATAC-seq data.

    Args:
        adata: AnnData object with ATAC counts
        binary: Whether to binarize the data
        normalize: Whether to normalize
        log_transform: Whether to apply log1p transformation
        tf_idf: Whether to apply TF-IDF transformation

    Returns:
        Preprocessed AnnData object
    """
    adata = adata.copy()

    # Store raw counts
    adata.layers['counts'] = adata.X.copy()

    if binary:
        # Binarize: any access > 0 becomes 1
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        adata.X = (X > 0).astype(np.float32)

    elif tf_idf:
        # TF-IDF transformation (common for ATAC-seq)
        from sklearn.feature_extraction.text import TfidfTransformer
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        tfidf = TfidfTransformer()
        adata.X = tfidf.fit_transform(X).toarray()

    else:
        # Standard normalization + log
        if normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)

        if log_transform:
            adata.X = np.log1p(adata.X)

    return adata


def select_highly_variable_genes(
    adata: AnnData,
    n_top_genes: int = 2000,
    flavor: str = 'seurat_v3',
    subset: bool = False,
) -> AnnData:
    """Select highly variable genes.

    Args:
        adata: AnnData object
        n_top_genes: Number of top genes to select
        flavor: Method for HVG selection ('seurat', 'cell_ranger', 'seurat_v3')
        subset: Whether to subset the data to HVGs

    Returns:
        AnnData object with HVG information
    """
    adata = adata.copy()
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        subset=subset
    )
    return adata


def batch_correction(
    adata: AnnData,
    batch_key: str,
    method: str = 'harmony',
    **kwargs
) -> AnnData:
    """Apply batch correction.

    Args:
        adata: AnnData object
        batch_key: Key in adata.obs containing batch information
        method: Batch correction method ('harmony', 'combat', 'scanorama')
        **kwargs: Additional arguments for the batch correction method

    Returns:
        Batch-corrected AnnData object
    """
    adata = adata.copy()

    if method == 'harmony':
        # Requires scanpy.external.pp.harmony_integrate
        import scanpy.external as sce
        # First compute PCA
        sc.pp.pca(adata)
        sce.pp.harmony_integrate(adata, batch_key, **kwargs)

    elif method == 'combat':
        sc.pp.combat(adata, key=batch_key, **kwargs)

    elif method == 'scanorama':
        # This requires splitting by batch and integrating
        # Simplified version - real implementation would be more complex
        raise NotImplementedError("Scanorama integration requires more setup")

    else:
        raise ValueError(f"Unknown batch correction method: {method}")

    return adata


def normalize_mudata(
    mdata: MuData,
    modality_params: Optional[dict] = None,
) -> MuData:
    """Normalize multi-modal data.

    Args:
        mdata: MuData object
        modality_params: Dictionary mapping modality names to preprocessing parameters

    Returns:
        Preprocessed MuData object
    """
    mdata = mdata.copy()

    default_params = {
        'rna': {'normalize': True, 'log_transform': True, 'n_top_genes': 2000},
        'protein': {'normalize': True, 'log_transform': True, 'clr_transform': False},
        'atac': {'normalize': True, 'log_transform': True, 'tf_idf': False},
    }

    if modality_params is None:
        modality_params = {}

    for mod_name in mdata.mod.keys():
        params = modality_params.get(mod_name, default_params.get(mod_name, {}))

        if mod_name == 'rna':
            mdata.mod[mod_name] = preprocess_rna(mdata.mod[mod_name], **params)
        elif mod_name == 'protein':
            mdata.mod[mod_name] = preprocess_protein(mdata.mod[mod_name], **params)
        elif mod_name == 'atac':
            mdata.mod[mod_name] = preprocess_atac(mdata.mod[mod_name], **params)

    return mdata
