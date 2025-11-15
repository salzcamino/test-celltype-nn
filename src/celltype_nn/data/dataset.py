"""Dataset classes for single-cell data."""

from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from anndata import AnnData
from mudata import MuData


class SingleCellDataset(Dataset):
    """PyTorch Dataset for single-cell RNA-seq data.

    Args:
        adata: AnnData object containing expression data
        label_key: Key in adata.obs containing cell type labels
        use_raw: Whether to use raw counts (adata.raw.X) or normalized (adata.X)
        gene_subset: Optional list of gene names to use as features
        transform: Optional transform to apply to the data
    """

    def __init__(
        self,
        adata: AnnData,
        label_key: str = "cell_type",
        use_raw: bool = False,
        gene_subset: Optional[List[str]] = None,
        transform: Optional[callable] = None,
    ):
        self.adata = adata
        self.label_key = label_key
        self.use_raw = use_raw
        self.transform = transform

        # Extract expression matrix
        if use_raw and adata.raw is not None:
            self.X = adata.raw.X
            gene_names = adata.raw.var_names
        else:
            self.X = adata.X
            gene_names = adata.var_names

        # Subset genes if specified
        if gene_subset is not None:
            gene_indices = [i for i, g in enumerate(gene_names) if g in gene_subset]
            if hasattr(self.X, 'toarray'):  # sparse matrix
                self.X = self.X[:, gene_indices]
            else:
                self.X = self.X[:, gene_indices]

        # Convert sparse to dense if needed
        if hasattr(self.X, 'toarray'):
            self.X = self.X.toarray()

        # Extract labels
        if label_key not in adata.obs.columns:
            raise ValueError(f"Label key '{label_key}' not found in adata.obs")

        self.labels = adata.obs[label_key].values

        # Create label encoding
        self.label_encoder = {label: idx for idx, label in enumerate(np.unique(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_encoder.items()}
        self.encoded_labels = np.array([self.label_encoder[label] for label in self.labels])

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return len(self.adata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single cell's data.

        Args:
            idx: Index of the cell

        Returns:
            Dictionary containing 'features' and 'label' tensors
        """
        # Get expression features
        features = self.X[idx].astype(np.float32)

        # Apply transform if specified
        if self.transform:
            features = self.transform(features)

        # Get label
        label = self.encoded_labels[idx]

        return {
            'features': torch.from_numpy(features),
            'label': torch.tensor(label, dtype=torch.long)
        }

    @property
    def num_features(self) -> int:
        """Return the number of features (genes)."""
        return self.X.shape[1]

    @property
    def num_classes(self) -> int:
        """Return the number of cell type classes."""
        return len(self.label_encoder)

    def get_label_name(self, idx: int) -> str:
        """Get the cell type name from encoded index."""
        return self.idx_to_label[idx]


class MultiModalDataset(Dataset):
    """PyTorch Dataset for multi-modal single-cell data (RNA + CITE-seq + ATAC-seq).

    Args:
        mdata: MuData object containing multi-modal data
        modalities: List of modality names to use (e.g., ['rna', 'protein', 'atac'])
        label_key: Key in mdata.obs containing cell type labels
        use_raw: Whether to use raw counts for each modality
        gene_subset: Optional dict mapping modality to list of feature names
        transform: Optional dict mapping modality to transform functions
    """

    def __init__(
        self,
        mdata: MuData,
        modalities: List[str] = ['rna'],
        label_key: str = "cell_type",
        use_raw: bool = False,
        gene_subset: Optional[Dict[str, List[str]]] = None,
        transform: Optional[Dict[str, callable]] = None,
    ):
        self.mdata = mdata
        self.modalities = modalities
        self.label_key = label_key
        self.use_raw = use_raw
        self.transform = transform or {}

        # Extract data for each modality
        self.data = {}
        for mod in modalities:
            if mod not in mdata.mod:
                raise ValueError(f"Modality '{mod}' not found in MuData object")

            adata = mdata.mod[mod]

            # Extract expression matrix
            if use_raw and adata.raw is not None:
                X = adata.raw.X
            else:
                X = adata.X

            # Subset features if specified
            if gene_subset and mod in gene_subset:
                feature_names = adata.var_names
                feature_indices = [i for i, f in enumerate(feature_names) if f in gene_subset[mod]]
                if hasattr(X, 'toarray'):
                    X = X[:, feature_indices]
                else:
                    X = X[:, feature_indices]

            # Convert sparse to dense
            if hasattr(X, 'toarray'):
                X = X.toarray()

            self.data[mod] = X

        # Extract labels
        if label_key not in mdata.obs.columns:
            raise ValueError(f"Label key '{label_key}' not found in mdata.obs")

        self.labels = mdata.obs[label_key].values

        # Create label encoding
        self.label_encoder = {label: idx for idx, label in enumerate(np.unique(self.labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_encoder.items()}
        self.encoded_labels = np.array([self.label_encoder[label] for label in self.labels])

    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return len(self.mdata)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Get a single cell's multi-modal data.

        Args:
            idx: Index of the cell

        Returns:
            Dictionary containing modality features and label
        """
        result = {}

        # Get features for each modality
        for mod in self.modalities:
            features = self.data[mod][idx].astype(np.float32)

            # Apply transform if specified
            if mod in self.transform:
                features = self.transform[mod](features)

            result[mod] = torch.from_numpy(features)

        # Get label
        result['label'] = torch.tensor(self.encoded_labels[idx], dtype=torch.long)

        return result

    @property
    def num_features(self) -> Dict[str, int]:
        """Return the number of features for each modality."""
        return {mod: data.shape[1] for mod, data in self.data.items()}

    @property
    def num_classes(self) -> int:
        """Return the number of cell type classes."""
        return len(self.label_encoder)

    def get_label_name(self, idx: int) -> str:
        """Get the cell type name from encoded index."""
        return self.idx_to_label[idx]
