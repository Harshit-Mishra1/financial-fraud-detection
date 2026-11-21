"""Evaluation and threshold-optimization utilities for imbalanced fraud detection."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(y_true, y_probs) -> tuple[float, float]:
    """Select the probability threshold that maximizes F1 score on the validation split."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_predictions(y_true, y_probs, threshold: float = 0.5) -> dict:
    """Compute fraud-relevant classification metrics at a specified decision threshold."""
    y_pred = (np.asarray(y_probs) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "pr_auc": float(average_precision_score(y_true, y_probs)),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
