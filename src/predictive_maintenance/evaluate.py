"""This module contains evaluation functions for the predictive maintenance."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """
    Evaluate model performance with metrics appropriate for a severely
    imbalanced classification problem.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        threshold: Decision threshold.

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "threshold": threshold,
    }


def find_threshold_for_recall(y_true, y_pred_proba, target_recall: float = 0.9) -> float:
    """
    Find the lowest decision threshold that achieves at least the target
    recall. Given the operational cost asymmetry (a missed failure costs
    far more than a false alarm), the threshold is chosen to favor recall
    rather than defaulting to the standard 0.5 cutoff.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities.
        target_recall: Target recall level.

    Returns:
        float: Decision threshold that achieves the target recall.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    valid = recalls[:-1] >= target_recall
    if not valid.any():
        return 0.5

    return float(np.max(thresholds[valid]))
