"""
Evaluation metrics for defect detection.
Supports: AUROC, AUPR, F1, F1-optimal, Accuracy, Pixel IoU.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _validate_inputs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    allow_single_class: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate metric inputs and return flattened numpy arrays.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted scores or predicted labels.
        allow_single_class: If True, skip the single-class check.
            This is useful for metrics like accuracy and fixed-threshold F1.

    Raises:
        ValueError: If arrays are empty, shapes mismatch, or y_true has only
            one class when allow_single_class is False.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if y_true.size == 0:
        raise ValueError("y_true is empty; cannot compute metric.")
    if y_score.size == 0:
        raise ValueError("y_score is empty; cannot compute metric.")
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_score has shape {y_score.shape}."
        )

    unique_classes = np.unique(y_true)
    if not allow_single_class and unique_classes.size < 2:
        raise ValueError(
            f"y_true contains only one class ({unique_classes[0]}); "
            "AUROC, AUPR, and optimal-threshold F1 are undefined on "
            "single-class data."
        )

    return y_true, y_score


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under ROC Curve."""
    y_true, y_score = _validate_inputs(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))


def compute_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area Under Precision-Recall Curve (a.k.a. Average Precision)."""
    y_true, y_score = _validate_inputs(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


# Backward-compat alias: older code calls compute_map. Keep until callers migrate.
compute_map = compute_aupr


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """F1 Score at a given threshold."""
    y_true, y_pred = _validate_inputs(
        y_true,
        y_pred,
        allow_single_class=True,
    )

    if np.issubdtype(y_pred.dtype, np.floating):
        y_pred = (y_pred >= threshold).astype(int)

    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_f1_optimal(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float]:
    """Sweep thresholds and return (best_f1, best_threshold).

    Uses the precision/recall pairs from sklearn's precision_recall_curve,
    computes F1 at each, and picks the maximum.
    """
    y_true, y_score = _validate_inputs(y_true, y_score)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # F1 = 2*P*R / (P + R); add epsilon to avoid divide-by-zero
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)

    # precision_recall_curve returns one extra precision/recall point with
    # no matching threshold, so drop the last F1 entry to align indices.
    f1_scores = f1_scores[:-1]

    if len(f1_scores) == 0:
        return 0.0, 0.5

    best_idx = int(np.argmax(f1_scores))
    return float(f1_scores[best_idx]), float(thresholds[best_idx])


def compute_accuracy(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    """Accuracy at a given threshold."""
    y_true, y_score = _validate_inputs(
        y_true,
        y_score,
        allow_single_class=True,
    )

    y_pred = (y_score >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


def compute_pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Pixel-wise Intersection over Union for segmentation.

    Args:
        pred_mask: Binary predicted mask [H, W]
        gt_mask: Binary ground truth mask [H, W]
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 1.0

    return float(intersection / union)


def evaluate_detector(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
    pred_masks: Optional[np.ndarray] = None,
    gt_masks: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Run the full evaluation suite.

    If `threshold` is None, the F1-optimal threshold is found automatically
    and used for F1 and accuracy. The chosen threshold is returned in the
    output dict under the key "threshold".

    Returns:
        dict with auroc, aupr, f1, accuracy, threshold,
        and optionally pixel_iou.
    """
    if threshold is None:
        f1_value, threshold = compute_f1_optimal(y_true, y_score)
    else:
        f1_value = compute_f1(y_true, y_score, threshold=threshold)

    aupr = compute_aupr(y_true, y_score)

    results: Dict[str, float] = {
        "auroc": compute_auroc(y_true, y_score),
        "aupr": aupr,
        "f1": f1_value,
        "accuracy": compute_accuracy(y_true, y_score, threshold=threshold),
        "threshold": float(threshold),
        # Backward-compat: older callers expected "map"
        "map": aupr,
    }

    if pred_masks is not None and gt_masks is not None:
        iou_scores = [
            compute_pixel_iou(p, g) for p, g in zip(pred_masks, gt_masks)
        ]
        results["pixel_iou"] = float(np.mean(iou_scores))

    return results


def print_results(results: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation results."""
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")

    for metric, value in results.items():
        print(f"  {metric.upper():<15} {value:.4f}")

    print(f"{'=' * 40}\n")