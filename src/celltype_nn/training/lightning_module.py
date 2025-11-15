"""PyTorch Lightning modules for training cell type classifiers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from typing import Optional, Dict, Any


class CellTypeClassifierModule(pl.LightningModule):
    """PyTorch Lightning module for cell type classification.

    Args:
        model: PyTorch model (e.g., RNAClassifier)
        num_classes: Number of cell type classes
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        scheduler: Learning rate scheduler ('cosine', 'step', 'plateau', None)
        class_weights: Optional class weights for handling imbalanced data
        use_focal_loss: Whether to use focal loss instead of cross-entropy
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer: str = 'adamw',
        scheduler: Optional[str] = 'cosine',
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
    ):
        super().__init__()

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler
        self.use_focal_loss = use_focal_loss

        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

        self.test_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.test_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        """Shared step for train/val/test."""
        features = batch['features']
        labels = batch['label']

        logits = self(features)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, preds, labels = self._shared_step(batch, batch_idx)

        # Update metrics
        self.train_acc(preds, labels)

        # Log
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, preds, labels = self._shared_step(batch, batch_idx)

        # Update metrics
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)

        # Log
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, preds, labels = self._shared_step(batch, batch_idx)

        # Update metrics
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)

        # Log
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        features = batch['features']
        logits = self(features)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        return {'predictions': preds, 'probabilities': probs}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        # Scheduler
        if self.scheduler_type is None:
            return optimizer

        elif self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }

        elif self.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }

        elif self.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch'
                }
            }

        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")


class MultiModalClassifierModule(pl.LightningModule):
    """PyTorch Lightning module for multi-modal cell type classification.

    Args:
        model: Multi-modal model
        num_classes: Number of cell type classes
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer: Optimizer type
        scheduler: Scheduler type
        class_weights: Optional class weights
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer: str = 'adamw',
        scheduler: Optional[str] = 'cosine',
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.scheduler_type = scheduler

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        """Forward pass with multi-modal input."""
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        """Shared step for train/val/test."""
        # Extract label
        labels = batch.pop('label')

        # Forward pass with remaining modalities
        logits = self(batch)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        # Put label back for next iteration
        batch['label'] = labels

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.train_acc(preds, labels)

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, preds, labels = self._shared_step(batch, batch_idx)

        self.val_acc(preds, labels)
        self.val_f1(preds, labels)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, preds, labels = self._shared_step(batch, batch_idx)

        self.test_acc(preds, labels)
        self.test_f1(preds, labels)

        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        if self.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        if self.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Reference: https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor for classes
        gamma: Focusing parameter
        weight: Manual rescaling weight given to each class
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)

        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
