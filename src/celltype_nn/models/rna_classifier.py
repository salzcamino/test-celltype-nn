"""RNA-based cell type classifier models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class RNAClassifier(nn.Module):
    """Deep feedforward neural network for cell type classification from RNA-seq.

    Args:
        input_dim: Number of input features (genes)
        num_classes: Number of cell type classes
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability
        batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'gelu', 'leaky_relu')
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
        activation: str = 'relu',
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self.activation)

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Encode
        features = self.encoder(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get learned embeddings without classification.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Embeddings of shape (batch_size, last_hidden_dim)
        """
        return self.encoder(x)


class RNAClassifierWithAttention(nn.Module):
    """RNA classifier with gene attention mechanism.

    This model learns to weight genes by importance for classification.

    Args:
        input_dim: Number of input features (genes)
        num_classes: Number of cell type classes
        hidden_dims: List of hidden layer dimensions
        attention_dim: Dimension of attention mechanism
        dropout_rate: Dropout probability
        batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        attention_dim: int = 128,
        dropout_rate: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Gene attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

        # Build encoder
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

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Compute attention weights for each gene
        # x: (batch_size, num_genes)
        # Reshape for attention: (batch_size, num_genes, 1)
        x_reshaped = x.unsqueeze(2)

        # Compute attention scores: (batch_size, num_genes, 1)
        attention_scores = self.attention(x_reshaped)

        # Apply softmax: (batch_size, num_genes, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention to features: (batch_size, num_genes)
        x_attended = x * attention_weights.squeeze(2)

        # Encode
        features = self.encoder(x_attended)

        # Classify
        logits = self.classifier(features)

        return logits

    def get_gene_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for gene importance.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Attention weights of shape (batch_size, input_dim)
        """
        x_reshaped = x.unsqueeze(2)
        attention_scores = self.attention(x_reshaped)
        attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights.squeeze(2)


class RNAVariationalAutoencoder(nn.Module):
    """Variational Autoencoder for RNA-seq data with cell type prediction.

    This model can be used for unsupervised pretraining before fine-tuning
    for classification.

    Args:
        input_dim: Number of input features (genes)
        num_classes: Number of cell type classes
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions for encoder/decoder
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Classifier on latent space
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x: torch.Tensor):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Returns:
            Tuple of (reconstruction, logits, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        logits = self.classifier(z)
        return reconstruction, logits, mu, logvar

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict cell type without reconstruction."""
        mu, _ = self.encode(x)
        logits = self.classifier(mu)
        return logits
