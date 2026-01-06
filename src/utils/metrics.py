"""
Metrics computation utilities.
"""

import torch
import numpy as np
from typing import Dict, Union, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    log_loss,
    confusion_matrix,
)


def compute_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    task_type: str = "classification",
    num_classes: int = 2,
    probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute metrics for classification or regression tasks.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task_type: "classification" or "regression"
        num_classes: Number of classes for classification
        probabilities: Class probabilities for AUC computation
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if probabilities is not None and isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    if task_type == "classification":
        return compute_classification_metrics(
            predictions, labels, num_classes, probabilities
        )
    else:
        return compute_regression_metrics(predictions, labels)


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 2,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        num_classes: Number of classes
        probabilities: Class probabilities for AUC computation
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(labels, predictions)
    
    # For binary classification
    if num_classes == 2:
        metrics["precision"] = precision_score(labels, predictions, zero_division=0)
        metrics["recall"] = recall_score(labels, predictions, zero_division=0)
        metrics["f1"] = f1_score(labels, predictions, zero_division=0)
        
        # AUC (requires probabilities)
        if probabilities is not None:
            try:
                if len(probabilities.shape) > 1:
                    probabilities = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.flatten()
                metrics["auc"] = roc_auc_score(labels, probabilities)
            except ValueError:
                metrics["auc"] = 0.0
    else:
        # Multi-class metrics
        metrics["precision_macro"] = precision_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(
            labels, predictions, average="macro", zero_division=0
        )
        
        metrics["precision_weighted"] = precision_score(
            labels, predictions, average="weighted", zero_division=0
        )
        metrics["recall_weighted"] = recall_score(
            labels, predictions, average="weighted", zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            labels, predictions, average="weighted", zero_division=0
        )
        
        # AUC for multi-class
        if probabilities is not None:
            try:
                metrics["auc_ovr"] = roc_auc_score(
                    labels, probabilities, multi_class="ovr", average="macro"
                )
            except ValueError:
                metrics["auc_ovr"] = 0.0
    
    return metrics


def compute_regression_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        predictions: Predicted values
        labels: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    metrics["mse"] = mean_squared_error(labels, predictions)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(labels, predictions)
    metrics["r2"] = r2_score(labels, predictions)
    
    # Mean Absolute Percentage Error
    mask = labels != 0
    if mask.any():
        metrics["mape"] = np.mean(np.abs((labels[mask] - predictions[mask]) / labels[mask])) * 100
    else:
        metrics["mape"] = 0.0
    
    return metrics


def get_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """
    Get confusion matrix.
    
    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        
    Returns:
        Confusion matrix
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    return confusion_matrix(labels.flatten(), predictions.flatten())
