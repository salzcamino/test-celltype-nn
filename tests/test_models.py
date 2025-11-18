"""Tests for neural network models."""

import pytest
import torch
import torch.nn as nn
from celltype_nn.models.rna_classifier import (
    RNAClassifier,
    RNAClassifierWithAttention,
    RNAVariationalAutoencoder,
)
from celltype_nn.models.multimodal_classifier import (
    MultiModalClassifier,
    MultiModalWithMissingModalities,
    ModalityEncoder,
)


class TestRNAClassifier:
    """Test RNAClassifier model."""

    def test_initialization(self):
        """Test model initialization."""
        model = RNAClassifier(
            input_dim=100,
            num_classes=5,
            hidden_dims=[64, 32],
            dropout_rate=0.3,
        )
        assert model.input_dim == 100
        assert model.num_classes == 5
        assert isinstance(model.encoder, nn.Sequential)
        assert isinstance(model.classifier, nn.Linear)

    def test_forward_pass(self):
        """Test forward pass."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        x = torch.randn(32, 100)
        logits = model(x)

        assert logits.shape == (32, 5)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_get_embeddings(self):
        """Test embedding extraction."""
        model = RNAClassifier(
            input_dim=100,
            num_classes=5,
            hidden_dims=[64, 32, 16]
        )
        x = torch.randn(32, 100)
        embeddings = model.get_embeddings(x)

        assert embeddings.shape == (32, 16)  # Last hidden dim

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ['relu', 'gelu', 'leaky_relu']:
            model = RNAClassifier(
                input_dim=100,
                num_classes=5,
                activation=activation
            )
            x = torch.randn(16, 100)
            logits = model(x)
            assert logits.shape == (16, 5)

    def test_without_batch_norm(self):
        """Test model without batch normalization."""
        model = RNAClassifier(
            input_dim=100,
            num_classes=5,
            batch_norm=False
        )
        x = torch.randn(8, 100)
        logits = model(x)
        assert logits.shape == (8, 5)

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        x = torch.randn(16, 100, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None


class TestRNAClassifierWithAttention:
    """Test RNAClassifierWithAttention model."""

    def test_initialization(self):
        """Test model initialization."""
        model = RNAClassifierWithAttention(
            input_dim=100,
            num_classes=5,
            attention_dim=64,
        )
        assert model.input_dim == 100
        assert model.num_classes == 5
        assert isinstance(model.attention, nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass with attention."""
        model = RNAClassifierWithAttention(input_dim=100, num_classes=5)
        x = torch.randn(32, 100)
        logits = model(x)

        assert logits.shape == (32, 5)

    def test_gene_importance(self):
        """Test gene importance extraction."""
        model = RNAClassifierWithAttention(input_dim=100, num_classes=5)
        x = torch.randn(32, 100)
        importance = model.get_gene_importance(x)

        assert importance.shape == (32, 100)
        # Attention weights should sum to 1
        assert torch.allclose(importance.sum(dim=1), torch.ones(32), atol=1e-5)
        # All weights should be positive
        assert (importance >= 0).all()


class TestRNAVariationalAutoencoder:
    """Test RNAVariationalAutoencoder model."""

    def test_initialization(self):
        """Test VAE initialization."""
        model = RNAVariationalAutoencoder(
            input_dim=100,
            num_classes=5,
            latent_dim=16,
        )
        assert model.input_dim == 100
        assert model.latent_dim == 16
        assert model.num_classes == 5

    def test_forward_pass(self):
        """Test VAE forward pass."""
        model = RNAVariationalAutoencoder(
            input_dim=100,
            num_classes=5,
            latent_dim=16,
        )
        x = torch.randn(32, 100)
        reconstruction, logits, mu, logvar = model(x)

        assert reconstruction.shape == (32, 100)
        assert logits.shape == (32, 5)
        assert mu.shape == (32, 16)
        assert logvar.shape == (32, 16)

    def test_encode(self):
        """Test encoding."""
        model = RNAVariationalAutoencoder(
            input_dim=100,
            num_classes=5,
            latent_dim=16,
        )
        x = torch.randn(32, 100)
        mu, logvar = model.encode(x)

        assert mu.shape == (32, 16)
        assert logvar.shape == (32, 16)

    def test_reparameterize(self):
        """Test reparameterization trick."""
        model = RNAVariationalAutoencoder(
            input_dim=100,
            num_classes=5,
            latent_dim=16,
        )
        mu = torch.randn(32, 16)
        logvar = torch.randn(32, 16)
        z = model.reparameterize(mu, logvar)

        assert z.shape == (32, 16)

    def test_predict(self):
        """Test prediction without reconstruction."""
        model = RNAVariationalAutoencoder(
            input_dim=100,
            num_classes=5,
            latent_dim=16,
        )
        x = torch.randn(32, 100)
        logits = model.predict(x)

        assert logits.shape == (32, 5)


class TestModalityEncoder:
    """Test ModalityEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ModalityEncoder(
            input_dim=100,
            output_dim=64,
            hidden_dims=[128, 96],
        )
        assert isinstance(encoder.encoder, nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass."""
        encoder = ModalityEncoder(input_dim=100, output_dim=64)
        x = torch.randn(32, 100)
        out = encoder(x)

        assert out.shape == (32, 64)


class TestMultiModalClassifier:
    """Test MultiModalClassifier."""

    def test_initialization_concat(self):
        """Test initialization with concat fusion."""
        model = MultiModalClassifier(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            embedding_dim=64,
            fusion_strategy='concat',
        )
        assert 'rna' in model.modality_encoders
        assert 'protein' in model.modality_encoders
        assert model.fusion_strategy == 'concat'

    def test_initialization_attention(self):
        """Test initialization with attention fusion."""
        model = MultiModalClassifier(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            embedding_dim=64,
            fusion_strategy='attention',
        )
        assert hasattr(model, 'attention_weights')

    def test_forward_pass_concat(self):
        """Test forward pass with concatenation."""
        model = MultiModalClassifier(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            embedding_dim=64,
            fusion_strategy='concat',
        )
        x = {
            'rna': torch.randn(32, 100),
            'protein': torch.randn(32, 50),
        }
        logits = model(x)

        assert logits.shape == (32, 5)

    def test_forward_pass_attention(self):
        """Test forward pass with attention fusion."""
        model = MultiModalClassifier(
            modality_dims={'rna': 100, 'protein': 50, 'atac': 200},
            num_classes=5,
            embedding_dim=64,
            fusion_strategy='attention',
        )
        x = {
            'rna': torch.randn(32, 100),
            'protein': torch.randn(32, 50),
            'atac': torch.randn(32, 200),
        }
        logits = model(x)

        assert logits.shape == (32, 5)

    def test_get_embeddings(self):
        """Test embedding extraction."""
        model = MultiModalClassifier(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            embedding_dim=64,
            fusion_strategy='concat',
        )
        x = {
            'rna': torch.randn(32, 100),
            'protein': torch.randn(32, 50),
        }
        embeddings = model.get_embeddings(x)

        # Concat: 2 modalities * 64 = 128
        assert embeddings.shape == (32, 128)


class TestMultiModalWithMissingModalities:
    """Test MultiModalWithMissingModalities."""

    def test_initialization(self):
        """Test initialization."""
        model = MultiModalWithMissingModalities(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            shared_dim=128,
        )
        assert model.shared_dim == 128
        assert len(model.modality_encoders) == 2

    def test_forward_all_modalities(self):
        """Test forward with all modalities present."""
        model = MultiModalWithMissingModalities(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            shared_dim=128,
        )
        x = {
            'rna': torch.randn(32, 100),
            'protein': torch.randn(32, 50),
        }
        logits = model(x)

        assert logits.shape == (32, 5)

    def test_forward_missing_modality(self):
        """Test forward with missing modality."""
        model = MultiModalWithMissingModalities(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            shared_dim=128,
        )
        # Only RNA present
        x = {
            'rna': torch.randn(32, 100),
        }
        logits = model(x)

        assert logits.shape == (32, 5)

    def test_forward_single_modality(self):
        """Test single modality forward."""
        model = MultiModalWithMissingModalities(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            shared_dim=128,
        )
        x = torch.randn(32, 100)
        logits = model.forward_single_modality(x, 'rna')

        assert logits.shape == (32, 5)

    def test_no_modalities_error(self):
        """Test that error is raised when no modalities present."""
        model = MultiModalWithMissingModalities(
            modality_dims={'rna': 100, 'protein': 50},
            num_classes=5,
            shared_dim=128,
        )
        with pytest.raises(ValueError):
            model({})


class TestModelIntegration:
    """Integration tests for models."""

    def test_all_models_backward_pass(self):
        """Test that all models support backpropagation."""
        models = [
            RNAClassifier(input_dim=50, num_classes=3),
            RNAClassifierWithAttention(input_dim=50, num_classes=3),
        ]

        for model in models:
            x = torch.randn(16, 50, requires_grad=True)
            logits = model(x)
            loss = logits.sum()
            loss.backward()

            # Check gradients exist
            for param in model.parameters():
                assert param.grad is not None

    def test_model_eval_mode(self):
        """Test that eval mode disables dropout."""
        model = RNAClassifier(
            input_dim=50,
            num_classes=3,
            dropout_rate=0.5
        )
        x = torch.randn(16, 50)

        # Train mode - might give different results
        model.train()
        out1 = model(x)

        # Eval mode - should give consistent results
        model.eval()
        out2 = model(x)
        out3 = model(x)

        assert torch.allclose(out2, out3)
