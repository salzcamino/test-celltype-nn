"""Evaluation metrics and visualization for cell type classification."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from typing import Optional, Dict, List
import torch


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for AUC)
        label_names: Names of cell type labels

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }

    # Per-class metrics
    if label_names is not None:
        f1_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)

        for i, label in enumerate(label_names):
            metrics[f'f1_{label}'] = f1_per_class[i]
            metrics[f'precision_{label}'] = precision_per_class[i]
            metrics[f'recall_{label}'] = recall_per_class[i]

    # ROC AUC if probabilities provided
    if y_prob is not None:
        try:
            # Multi-class ROC AUC
            n_classes = y_prob.shape[1]
            if n_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='weighted'
                )
        except Exception as e:
            print(f"Could not compute ROC AUC: {e}")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
):
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names of cell type labels
        normalize: Whether to normalize counts
        figsize: Figure size
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
):
    """Plot per-class F1, precision, and recall.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names of cell type labels
        figsize: Figure size
        save_path: Path to save figure
    """
    f1 = f1_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    x = np.arange(len(label_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, f1, width, label='F1')
    ax.bar(x, precision, width, label='Precision')
    ax.bar(x + width, recall, width, label='Recall')

    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
):
    """Plot ROC curves for each class.

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        label_names: Names of cell type labels
        figsize: Figure size
        save_path: Path to save figure
    """
    from sklearn.preprocessing import label_binarize

    n_classes = len(label_names)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=figsize)

    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])

        plt.plot(fpr, tpr, label=f'{label_names[i]} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names of cell type labels
        output_path: Path to save report

    Returns:
        Classification report as string
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=4
    )

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


def evaluate_model(
    model,
    dataloader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Evaluate a trained model on a dataset.

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        label_names: Names of cell type labels

    Returns:
        Dictionary containing predictions, probabilities, and metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Handle both single-modal and multi-modal batches
            if isinstance(batch, dict) and 'features' in batch:
                features = batch['features'].to(device)
                labels = batch['label']
                logits = model(features)
            elif isinstance(batch, dict) and 'label' in batch:
                # Multi-modal case
                labels = batch.pop('label')
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch)
                batch['label'] = labels
            else:
                raise ValueError("Unexpected batch format")

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate results
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, label_names)

    return {
        'predictions': y_pred,
        'probabilities': y_prob,
        'labels': y_true,
        'metrics': metrics
    }
