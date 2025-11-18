"""Integration tests for complete workflows."""

import pytest
import torch
import tempfile
from pathlib import Path
from celltype_nn.models.rna_classifier import RNAClassifier
from celltype_nn.training.lightning_module import CellTypeClassifierModule
from celltype_nn.data.dataset import SingleCellDataset
from celltype_nn.data.loader import create_dataloaders
from celltype_nn.preprocessing.preprocess import preprocess_rna


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_training_workflow(self, larger_adata):
        """Test complete training pipeline."""
        # 1. Preprocess data
        adata = preprocess_rna(
            larger_adata,
            min_genes=50,
            min_cells=3,
            n_top_genes=500,
            normalize=True,
            log_transform=True,
        )

        # Subset to HVGs
        adata = adata[:, adata.var['highly_variable']].copy()

        # 2. Create dataloaders
        dataloaders = create_dataloaders(
            adata,
            label_key='cell_type',
            batch_size=64,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
        )

        # 3. Get dataset info
        train_dataset = dataloaders['datasets']['train']
        n_features = train_dataset.num_features
        n_classes = train_dataset.num_classes

        # 4. Create model
        model = RNAClassifier(
            input_dim=n_features,
            num_classes=n_classes,
            hidden_dims=[256, 128],
            dropout_rate=0.3,
        )

        # 5. Create Lightning module
        lightning_module = CellTypeClassifierModule(
            model=model,
            num_classes=n_classes,
            learning_rate=0.001,
        )

        # 6. Run a few training steps
        train_losses = []
        for i, batch in enumerate(dataloaders['train']):
            if i >= 5:  # Just a few iterations
                break
            loss = lightning_module.training_step(batch, i)
            train_losses.append(loss.item())

        # All losses should be positive and finite
        assert all(loss > 0 for loss in train_losses)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in train_losses)

    def test_prediction_workflow(self, larger_adata):
        """Test complete prediction pipeline."""
        # 1. Preprocess
        adata = preprocess_rna(
            larger_adata,
            min_genes=50,
            min_cells=3,
            n_top_genes=500,
            normalize=True,
            log_transform=True,
        )
        adata = adata[:, adata.var['highly_variable']].copy()

        # 2. Create dataset
        dataset = SingleCellDataset(adata, label_key='cell_type')

        # 3. Create a simple trained model (just initialized for testing)
        model = RNAClassifier(
            input_dim=dataset.num_features,
            num_classes=dataset.num_classes,
            hidden_dims=[128, 64],
        )
        model.eval()

        # 4. Make predictions
        predictions = []
        with torch.no_grad():
            for i in range(min(10, len(dataset))):
                sample = dataset[i]
                features = sample['features'].unsqueeze(0)
                logits = model(features)
                pred = torch.argmax(logits, dim=1)
                predictions.append(pred.item())

        # Should have predictions
        assert len(predictions) > 0
        # All predictions should be valid class indices
        assert all(0 <= p < dataset.num_classes for p in predictions)

    def test_data_consistency_across_splits(self, larger_adata):
        """Test that data splits are consistent."""
        adata = preprocess_rna(larger_adata, n_top_genes=500)
        adata = adata[:, adata.var['highly_variable']].copy()

        dataloaders = create_dataloaders(
            adata,
            label_key='cell_type',
            batch_size=32,
            random_state=42,  # Fixed seed
        )

        train_ds = dataloaders['datasets']['train']
        val_ds = dataloaders['datasets']['val']
        test_ds = dataloaders['datasets']['test']

        # All datasets should have same features and classes
        assert train_ds.num_features == val_ds.num_features == test_ds.num_features
        assert train_ds.num_classes == val_ds.num_classes == test_ds.num_classes

        # Label encoders should match
        assert train_ds.label_encoder == val_ds.label_encoder
        assert train_ds.label_encoder == test_ds.label_encoder

    def test_model_save_load(self, tmp_path):
        """Test saving and loading models."""
        # Create a model
        model = RNAClassifier(
            input_dim=100,
            num_classes=5,
            hidden_dims=[64, 32],
        )

        # Make a prediction
        x = torch.randn(1, 100)
        model.eval()
        with torch.no_grad():
            original_output = model(x)

        # Save model state
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        # Create new model and load state
        new_model = RNAClassifier(
            input_dim=100,
            num_classes=5,
            hidden_dims=[64, 32],
        )
        new_model.load_state_dict(torch.load(save_path))
        new_model.eval()

        # Should get same output
        with torch.no_grad():
            loaded_output = new_model(x)

        assert torch.allclose(original_output, loaded_output)

    def test_batch_processing(self, larger_adata):
        """Test processing data in batches."""
        adata = preprocess_rna(larger_adata, n_top_genes=500)
        adata = adata[:, adata.var['highly_variable']].copy()

        dataset = SingleCellDataset(adata, label_key='cell_type')

        model = RNAClassifier(
            input_dim=dataset.num_features,
            num_classes=dataset.num_classes,
        )
        model.eval()

        # Process in batches
        batch_size = 32
        all_predictions = []

        for i in range(0, min(100, len(dataset)), batch_size):
            batch_features = []
            for j in range(i, min(i + batch_size, len(dataset))):
                sample = dataset[j]
                batch_features.append(sample['features'])

            batch = torch.stack(batch_features)

            with torch.no_grad():
                logits = model(batch)
                preds = torch.argmax(logits, dim=1)
                all_predictions.extend(preds.tolist())

        assert len(all_predictions) > 0

    def test_different_batch_sizes(self, larger_adata):
        """Test that model works with different batch sizes."""
        model = RNAClassifier(input_dim=500, num_classes=5)
        model.eval()

        batch_sizes = [1, 4, 16, 32, 64]

        for bs in batch_sizes:
            x = torch.randn(bs, 500)
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (bs, 5)

    def test_reproducibility_with_seed(self, simple_adata):
        """Test that results are reproducible with fixed seed."""
        torch.manual_seed(42)

        model1 = RNAClassifier(input_dim=50, num_classes=3)
        x = torch.randn(16, 50)

        torch.manual_seed(42)
        model2 = RNAClassifier(input_dim=50, num_classes=3)

        # Models initialized with same seed should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

        # Should produce same outputs
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        assert torch.equal(out1, out2)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_wrong_input_dimension(self):
        """Test error with wrong input dimension."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        x = torch.randn(16, 50)  # Wrong dimension

        with pytest.raises(RuntimeError):
            model(x)

    def test_empty_batch_error(self):
        """Test handling of empty inputs."""
        model = RNAClassifier(input_dim=100, num_classes=5)
        x = torch.randn(0, 100)  # Empty batch

        # This might work (producing empty output) or raise error
        # depending on PyTorch version
        try:
            logits = model(x)
            assert logits.shape == (0, 5)
        except RuntimeError:
            pass  # Also acceptable

    def test_invalid_activation_error(self):
        """Test error with invalid activation."""
        with pytest.raises(ValueError):
            RNAClassifier(
                input_dim=100,
                num_classes=5,
                activation='invalid_activation'
            )
