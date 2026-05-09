"""Tests for evaluation metrics."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.evaluation.metrics import (
    compute_accuracy,
    compute_aupr,
    compute_auroc,
    compute_f1,
    compute_f1_optimal,
    compute_map,
    compute_pixel_iou,
    evaluate_detector,
)


# ──────────────────────────────────────────────────────────────────────
# AUROC
# ──────────────────────────────────────────────────────────────────────


def test_auroc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auroc(y_true, y_score) == 1.0


def test_auroc_random():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    auroc = compute_auroc(y_true, y_score)
    assert 0.4 <= auroc <= 0.6


def test_auroc_inverted():
    """If we flip scores, AUROC should flip too (1 - original)."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_auroc(y_true, y_score) == 0.0


def test_auroc_sklearn_parity():
    """compute_auroc must agree with sklearn on random arrays."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_true = rng.integers(0, 2, size=100)
        y_score = rng.uniform(size=100)
        if len(np.unique(y_true)) < 2:
            continue  # skip degenerate cases
        ours = compute_auroc(y_true, y_score)
        theirs = roc_auc_score(y_true, y_score)
        assert abs(ours - theirs) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# AUPR
# ──────────────────────────────────────────────────────────────────────


def test_aupr_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_aupr(y_true, y_score) == 1.0


def test_aupr_sklearn_parity():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y_true = rng.integers(0, 2, size=100)
        y_score = rng.uniform(size=100)
        if len(np.unique(y_true)) < 2:
            continue
        ours = compute_aupr(y_true, y_score)
        theirs = average_precision_score(y_true, y_score)
        assert abs(ours - theirs) < 1e-9


def test_compute_map_alias():
    """compute_map should still exist as a backward-compat alias for AUPR."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_map(y_true, y_score) == compute_aupr(y_true, y_score)


# ──────────────────────────────────────────────────────────────────────
# F1
# ──────────────────────────────────────────────────────────────────────


def test_f1_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_f1(y_true, y_pred, threshold=0.5) == 1.0


def test_f1_sklearn_parity():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=100)
    y_pred_int = rng.integers(0, 2, size=100)
    ours = compute_f1(y_true, y_pred_int.astype(float), threshold=0.5)
    theirs = f1_score(y_true, (y_pred_int >= 0.5).astype(int), zero_division=0)
    assert abs(ours - theirs) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# F1-optimal
# ──────────────────────────────────────────────────────────────────────


def test_f1_optimal_finds_best_threshold():
    """On a hand-built case, 0.5 should NOT be optimal — 0.6 should be."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.55, 0.65, 0.8, 0.9])
    best_f1, best_thr = compute_f1_optimal(y_true, y_score)
    # The best split lies between 0.55 and 0.65
    assert best_f1 == pytest.approx(1.0)
    assert 0.55 < best_thr <= 0.65


def test_f1_optimal_returns_floats():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.7, 0.3, 0.8])
    best_f1, best_thr = compute_f1_optimal(y_true, y_score)
    assert isinstance(best_f1, float)
    assert isinstance(best_thr, float)


# ──────────────────────────────────────────────────────────────────────
# Accuracy
# ──────────────────────────────────────────────────────────────────────


def test_accuracy_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_accuracy(y_true, y_score, threshold=0.5) == 1.0


def test_accuracy_sklearn_parity():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=100)
    y_score = rng.uniform(size=100)
    ours = compute_accuracy(y_true, y_score, threshold=0.5)
    theirs = accuracy_score(y_true, (y_score >= 0.5).astype(int))
    assert abs(ours - theirs) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# Pixel IoU
# ──────────────────────────────────────────────────────────────────────


def test_pixel_iou_perfect():
    mask = np.array([[1, 0], [0, 1]])
    assert compute_pixel_iou(mask, mask) == 1.0


def test_pixel_iou_no_overlap():
    pred = np.array([[1, 0], [0, 0]])
    gt = np.array([[0, 1], [0, 0]])
    assert compute_pixel_iou(pred, gt) == 0.0


# ──────────────────────────────────────────────────────────────────────
# evaluate_detector
# ──────────────────────────────────────────────────────────────────────


def test_evaluate_detector_returns_all_keys():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    results = evaluate_detector(y_true, y_score)
    for key in ("auroc", "aupr", "f1", "accuracy", "threshold", "map"):
        assert key in results
    assert results["auroc"] == 1.0


def test_evaluate_detector_uses_optimal_threshold_when_none():
    """When threshold=None, evaluator should pick the F1-optimal threshold."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.55, 0.65, 0.8, 0.9])
    results = evaluate_detector(y_true, y_score, threshold=None)
    assert results["f1"] == pytest.approx(1.0)
    assert 0.55 < results["threshold"] <= 0.65


def test_evaluate_detector_respects_explicit_threshold():
    """When threshold is given explicitly, it must be used as-is."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    results = evaluate_detector(y_true, y_score, threshold=0.5)
    assert results["threshold"] == 0.5