"""Tests for evaluation metrics."""

import pytest
import numpy as np
from celltype_nn.evaluation.metrics import (
    compute_metrics,
    generate_classification_report,
)


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_basic_metrics(self):
        """Test basic metric computation."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])  # Perfect predictions

        metrics = compute_metrics(y_true, y_pred)

        assert metrics['accuracy'] == 1.0
        assert metrics['f1_macro'] == 1.0
        assert metrics['f1_micro'] == 1.0
        assert metrics['precision_macro'] == 1.0
        assert metrics['recall_macro'] == 1.0

    def test_imperfect_predictions(self):
        """Test with imperfect predictions."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])  # Some errors

        metrics = compute_metrics(y_true, y_pred)

        assert 0 < metrics['accuracy'] < 1
        assert 0 < metrics['f1_macro'] < 1

    def test_with_probabilities(self):
        """Test with probability predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        y_prob = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        # Should have ROC AUC metrics
        assert 'roc_auc_macro' in metrics or 'roc_auc' in metrics

    def test_with_label_names(self):
        """Test with label names."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        label_names = ['TypeA', 'TypeB', 'TypeC']

        metrics = compute_metrics(y_true, y_pred, label_names=label_names)

        # Should have per-class metrics
        assert 'f1_TypeA' in metrics
        assert 'f1_TypeB' in metrics
        assert 'f1_TypeC' in metrics
        assert 'precision_TypeA' in metrics
        assert 'recall_TypeA' in metrics

    def test_binary_classification(self):
        """Test binary classification metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.3, 0.7],
        ])

        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert 'roc_auc' in metrics
        assert metrics['roc_auc'] == 1.0  # Perfect predictions


class TestClassificationReport:
    """Test classification report generation."""

    def test_generate_report(self):
        """Test generating classification report."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        label_names = ['TypeA', 'TypeB', 'TypeC']

        report = generate_classification_report(y_true, y_pred, label_names)

        assert isinstance(report, str)
        assert 'TypeA' in report
        assert 'TypeB' in report
        assert 'TypeC' in report
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report

    def test_report_without_labels(self):
        """Test report without label names."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        report = generate_classification_report(y_true, y_pred)

        assert isinstance(report, str)
        assert 'precision' in report
