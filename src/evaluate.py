"""Evaluation helpers. Call these after fitting a model."""

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, y_pred, y_score=None) -> dict:
    """Return common classification metrics.

    y_pred should be predicted class labels; y_score should be fraud probabilities
    or decision scores for the positive (fraud) class.
    """
    result = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
    }
    if y_score is not None:
        result["pr_auc_average_precision"] = average_precision_score(y_true, y_score)
    return result
