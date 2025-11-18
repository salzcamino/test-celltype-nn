"""Tests for training modules."""

import pytest
import torch
import torch.nn as nn
from celltype_nn.models.rna_classifier import RNAClassifier
from celltype_nn.training.lightning_module import CellTypeClassifierModule, FocalLoss


class TestCellTypeClassifierModule:
    """Test CellTypeClassifierModule."""

    def test_initialization(self):
        """Test module initialization."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
            learning_rate=0.001,
        )

        assert lightning_module.model == model
        assert lightning_module.num_classes == 5
        assert lightning_module.learning_rate == 0.001

    def test_forward_pass(self):
        """Test forward pass."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
        )

        x = torch.randn(16, 100)
        logits = lightning_module(x)

        assert logits.shape == (16, 5)

    def test_training_step(self):
        """Test training step."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
        )

        batch = {
            'features': torch.randn(16, 100),
            'label': torch.randint(0, 5, (16,)),
        }

        loss = lightning_module.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss > 0

    def test_validation_step(self):
        """Test validation step."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
        )

        batch = {
            'features': torch.randn(16, 100),
            'label': torch.randint(0, 5, (16,)),
        }

        loss = lightning_module.validation_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss > 0

    def test_configure_optimizers_adam(self):
        """Test optimizer configuration with Adam."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
            optimizer='adam',
            scheduler=None,
        )

        optimizer = lightning_module.configure_optimizers()

        assert isinstance(optimizer, torch.optim.Adam)

    def test_configure_optimizers_adamw(self):
        """Test optimizer configuration with AdamW."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
            optimizer='adamw',
            scheduler=None,
        )

        optimizer = lightning_module.configure_optimizers()

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_predict_step(self):
        """Test prediction step."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
        )
        lightning_module.eval()

        batch = {
            'features': torch.randn(16, 100),
        }

        result = lightning_module.predict_step(batch, 0)

        assert 'predictions' in result
        assert 'probabilities' in result
        assert result['predictions'].shape == (16,)
        assert result['probabilities'].shape == (16, 5)
        # Probabilities should sum to 1
        assert torch.allclose(
            result['probabilities'].sum(dim=1),
            torch.ones(16),
            atol=1e-5
        )


class TestFocalLoss:
    """Test Focal Loss implementation."""

    def test_initialization(self):
        """Test focal loss initialization."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0

    def test_forward_pass(self):
        """Test focal loss forward pass."""
        loss_fn = FocalLoss()

        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss > 0

    def test_focal_vs_ce_loss(self):
        """Test that focal loss differs from CE loss."""
        focal_loss = FocalLoss(gamma=2.0)
        ce_loss = nn.CrossEntropyLoss()

        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        fl = focal_loss(logits, targets)
        cel = ce_loss(logits, targets)

        # They should be different (unless gamma=0, but we use gamma=2)
        assert not torch.allclose(fl, cel)

    def test_gradient_flow(self):
        """Test that gradients flow through focal loss."""
        loss_fn = FocalLoss()

        logits = torch.randn(16, 5, requires_grad=True)
        targets = torch.randint(0, 5, (16,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestTrainingIntegration:
    """Integration tests for training."""

    def test_full_training_iteration(self):
        """Test a full training iteration."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
            learning_rate=0.001,
        )

        # Training batch
        train_batch = {
            'features': torch.randn(32, 100),
            'label': torch.randint(0, 5, (32,)),
        }

        # Forward pass
        loss = lightning_module.training_step(train_batch, 0)
        assert loss > 0

        # Validation batch
        val_batch = {
            'features': torch.randn(16, 100),
            'label': torch.randint(0, 5, (16,)),
        }

        val_loss = lightning_module.validation_step(val_batch, 0)
        assert val_loss > 0

    def test_optimizer_step(self):
        """Test that optimizer updates parameters."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=5,
            learning_rate=0.01,
            optimizer='sgd',
            scheduler=None,
        )

        optimizer = lightning_module.configure_optimizers()

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Training step
        batch = {
            'features': torch.randn(32, 100),
            'label': torch.randint(0, 5, (32,)),
        }

        loss = lightning_module.training_step(batch, 0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Parameters should have changed
        for initial, current in zip(initial_params, model.parameters()):
            assert not torch.equal(initial, current)
