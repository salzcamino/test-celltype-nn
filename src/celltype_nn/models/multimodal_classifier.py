"""Multi-modal cell type classifier models."""

import torch
import torch.nn as nn
from typing import Optional, List, Dict


class ModalityEncoder(nn.Module):
    """Encoder for a single modality.

    Args:
        input_dim: Number of input features
        output_dim: Dimension of output embedding
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
        batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Final projection to output dimension
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MultiModalClassifier(nn.Module):
    """Multi-modal classifier with late fusion strategy.

    Each modality is encoded separately, then embeddings are concatenated
    and passed through a classification head.

    Args:
        modality_dims: Dictionary mapping modality names to input dimensions
        num_classes: Number of cell type classes
        embedding_dim: Dimension of each modality's embedding
        hidden_dims: Hidden dimensions for modality encoders
        classifier_dims: Hidden dimensions for final classifier
        dropout_rate: Dropout probability
        fusion_strategy: How to fuse modalities ('concat', 'attention')
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        classifier_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        fusion_strategy: str = 'concat',
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        if classifier_dims is None:
            classifier_dims = [128, 64]

        self.modality_names = list(modality_dims.keys())
        self.embedding_dim = embedding_dim
        self.fusion_strategy = fusion_strategy

        # Create encoder for each modality
        self.modality_encoders = nn.ModuleDict()
        for modality, input_dim in modality_dims.items():
            self.modality_encoders[modality] = ModalityEncoder(
                input_dim=input_dim,
                output_dim=embedding_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate
            )

        # Fusion layer
        if fusion_strategy == 'concat':
            fused_dim = embedding_dim * len(modality_dims)
        elif fusion_strategy == 'attention':
            # Attention-based fusion
            self.attention_weights = nn.Linear(embedding_dim, 1)
            fused_dim = embedding_dim
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # Classification head
        classifier_layers = []
        prev_dim = fused_dim

        for hidden_dim in classifier_dims:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with multi-modal input.

        Args:
            x: Dictionary mapping modality names to input tensors

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Encode each modality
        embeddings = {}
        for modality in self.modality_names:
            if modality in x:
                embeddings[modality] = self.modality_encoders[modality](x[modality])

        # Fuse embeddings
        if self.fusion_strategy == 'concat':
            # Concatenate all embeddings
            fused = torch.cat([embeddings[mod] for mod in self.modality_names if mod in embeddings], dim=1)

        elif self.fusion_strategy == 'attention':
            # Stack embeddings: (batch_size, num_modalities, embedding_dim)
            stacked = torch.stack([embeddings[mod] for mod in self.modality_names if mod in embeddings], dim=1)

            # Compute attention weights: (batch_size, num_modalities, 1)
            attention_scores = self.attention_weights(stacked)
            attention_weights = torch.softmax(attention_scores, dim=1)

            # Weighted sum: (batch_size, embedding_dim)
            fused = (stacked * attention_weights).sum(dim=1)

        # Classify
        logits = self.classifier(fused)

        return logits

    def get_embeddings(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get fused embeddings without classification.

        Args:
            x: Dictionary mapping modality names to input tensors

        Returns:
            Fused embeddings
        """
        embeddings = {}
        for modality in self.modality_names:
            if modality in x:
                embeddings[modality] = self.modality_encoders[modality](x[modality])

        if self.fusion_strategy == 'concat':
            fused = torch.cat([embeddings[mod] for mod in self.modality_names if mod in embeddings], dim=1)
        elif self.fusion_strategy == 'attention':
            stacked = torch.stack([embeddings[mod] for mod in self.modality_names if mod in embeddings], dim=1)
            attention_scores = self.attention_weights(stacked)
            attention_weights = torch.softmax(attention_scores, dim=1)
            fused = (stacked * attention_weights).sum(dim=1)

        return fused


class MultiModalWithMissingModalities(nn.Module):
    """Multi-modal classifier that handles missing modalities gracefully.

    Uses modality-specific parameters with a shared embedding space.

    Args:
        modality_dims: Dictionary mapping modality names to input dimensions
        num_classes: Number of cell type classes
        shared_dim: Dimension of shared embedding space
        modality_specific_dims: Hidden dims for modality encoders
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        shared_dim: int = 128,
        modality_specific_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if modality_specific_dims is None:
            modality_specific_dims = [256, 128]

        self.modality_names = list(modality_dims.keys())
        self.shared_dim = shared_dim

        # Create encoder for each modality that projects to shared space
        self.modality_encoders = nn.ModuleDict()
        for modality, input_dim in modality_dims.items():
            self.modality_encoders[modality] = ModalityEncoder(
                input_dim=input_dim,
                output_dim=shared_dim,
                hidden_dims=modality_specific_dims,
                dropout_rate=dropout_rate
            )

        # Shared classifier (works on shared embedding space)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Learnable aggregation weights for when multiple modalities present
        self.modality_weights = nn.Parameter(torch.ones(len(modality_dims)))

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass handling missing modalities.

        Args:
            x: Dictionary mapping available modality names to input tensors

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        embeddings = []
        weights = []

        for i, modality in enumerate(self.modality_names):
            if modality in x and x[modality] is not None:
                # Encode modality to shared space
                emb = self.modality_encoders[modality](x[modality])
                embeddings.append(emb)
                weights.append(self.modality_weights[i])

        if len(embeddings) == 0:
            raise ValueError("At least one modality must be present")

        # Weighted average of embeddings
        weights = torch.stack(weights)
        weights = torch.softmax(weights, dim=0)

        stacked_embeddings = torch.stack(embeddings, dim=0)  # (num_present, batch_size, shared_dim)
        weights = weights.view(-1, 1, 1)  # (num_present, 1, 1)

        # Aggregate embeddings
        aggregated = (stacked_embeddings * weights).sum(dim=0)  # (batch_size, shared_dim)

        # Classify
        logits = self.classifier(aggregated)

        return logits

    def forward_single_modality(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Forward pass for a single modality.

        Useful for inference when only one modality is available.

        Args:
            x: Input tensor for the modality
            modality: Name of the modality

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        if modality not in self.modality_encoders:
            raise ValueError(f"Unknown modality: {modality}")

        embedding = self.modality_encoders[modality](x)
        logits = self.classifier(embedding)
        return logits
