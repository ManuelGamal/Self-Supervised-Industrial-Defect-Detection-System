"""
Bootstrap confidence intervals for evaluation metrics.

Provides:
- bootstrap_ci: 95% CI for any single-array metric (e.g. AUROC)
- paired_bootstrap_diff: 95% CI for the difference between two predictors
  evaluated on the same samples (used for fold-vs-fold comparisons)
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute a sample-level bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true:
        Ground-truth labels, shape (N,).
    y_score:
        Predicted scores or probabilities, shape (N,).
    metric_fn:
        Function with signature ``metric_fn(y_true, y_score) -> float``,
        e.g. ``compute_auroc`` or ``compute_aupr``.
    n_resamples:
        Number of bootstrap resamples. 10,000 is a typical default.
    confidence:
        Confidence level in (0, 1). 0.95 means 95% CI.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    point_estimate, ci_low, ci_high
        Point estimate is computed on the original data; the CI bounds
        are the (1 - confidence)/2 and (1 + confidence)/2 percentiles of
        the bootstrap distribution.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        raise ValueError("y_true is empty; cannot bootstrap.")
    if len(y_score) != n:
        raise ValueError("y_true and y_score must have the same length.")

    rng = np.random.default_rng(seed)
    point_estimate = float(metric_fn(y_true, y_score))

    boot_scores = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        try:
            boot_scores[i] = metric_fn(y_true[idx], y_score[idx])
        except ValueError:
            # Some metrics (e.g. AUROC) fail when a resample has only
            # one class. Treat that resample as missing.
            boot_scores[i] = np.nan

    boot_scores = boot_scores[~np.isnan(boot_scores)]
    if len(boot_scores) == 0:
        # Degenerate case: every resample was single-class.
        return point_estimate, point_estimate, point_estimate

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_scores, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return point_estimate, ci_low, ci_high


def paired_bootstrap_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Paired bootstrap CI for ``metric(A) - metric(B)`` on the same samples.

    Both predictors are evaluated on the *same* resampled indices, which
    correctly accounts for the correlation between them. Use this for
    fold-vs-fold or model-vs-model comparisons on a shared test set.

    Returns ``(diff_point, diff_low, diff_high)``. If the CI does not
    contain zero, the difference is statistically significant at the
    chosen confidence level.
    """
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    n = len(y_true)
    if not (len(y_score_a) == n == len(y_score_b)):
        raise ValueError("All inputs must have the same length.")

    rng = np.random.default_rng(seed)
    diff_point = float(metric_fn(y_true, y_score_a) - metric_fn(y_true, y_score_b))

    diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            metric_a = metric_fn(y_true[idx], y_score_a[idx])
            metric_b = metric_fn(y_true[idx], y_score_b[idx])
            diffs[i] = metric_a - metric_b
        except ValueError:
            diffs[i] = np.nan

    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return diff_point, diff_point, diff_point

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(diffs, 100 * alpha / 2))
    ci_high = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return diff_point, ci_low, ci_high