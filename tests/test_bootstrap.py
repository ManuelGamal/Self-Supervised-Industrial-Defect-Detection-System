"""Tests for bootstrap confidence intervals."""

import numpy as np
import pytest

from src.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_diff
from src.evaluation.metrics import compute_auroc, compute_aupr


# ──────────────────────────────────────────────────────────────────────
# bootstrap_ci
# ──────────────────────────────────────────────────────────────────────


def test_bootstrap_ci_brackets_point_estimate():
    """The CI must contain the point estimate by construction."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_score = y_true + rng.normal(0, 0.5, size=200)
    point, low, high = bootstrap_ci(
        y_true, y_score, compute_auroc, n_resamples=1000, seed=0
    )
    assert low <= point <= high


def test_bootstrap_ci_width_is_positive():
    """For non-degenerate inputs the CI should have non-zero width."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200)
    y_score = rng.uniform(size=200)
    _, low, high = bootstrap_ci(
        y_true, y_score, compute_auroc, n_resamples=1000, seed=1
    )
    assert high > low


def test_bootstrap_ci_reproducible_with_same_seed():
    """Same seed must give exactly the same CI bounds."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=100)
    y_score = rng.uniform(size=100)

    out_a = bootstrap_ci(y_true, y_score, compute_auroc,
                         n_resamples=500, seed=42)
    out_b = bootstrap_ci(y_true, y_score, compute_auroc,
                         n_resamples=500, seed=42)
    assert out_a == out_b


def test_bootstrap_ci_works_with_aupr():
    """bootstrap_ci is metric-agnostic — try it with AUPR too."""
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=150)
    y_score = y_true + rng.normal(0, 0.4, size=150)
    point, low, high = bootstrap_ci(
        y_true, y_score, compute_aupr, n_resamples=500, seed=3
    )
    assert 0.0 <= low <= point <= high <= 1.0


def test_bootstrap_ci_raises_on_empty_input():
    with pytest.raises(ValueError):
        bootstrap_ci(
            np.array([]), np.array([]), compute_auroc, n_resamples=10
        )


def test_bootstrap_ci_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        bootstrap_ci(
            np.array([0, 1]), np.array([0.5]),
            compute_auroc, n_resamples=10,
        )


def test_bootstrap_ci_narrows_with_more_samples():
    """Larger sample size should produce a tighter CI on average."""
    rng = np.random.default_rng(4)

    # Small dataset
    y_true_small = rng.integers(0, 2, size=50)
    y_score_small = y_true_small + rng.normal(0, 0.5, size=50)
    _, lo_s, hi_s = bootstrap_ci(
        y_true_small, y_score_small, compute_auroc,
        n_resamples=1000, seed=4,
    )
    width_small = hi_s - lo_s

    # Larger dataset, same signal-to-noise ratio
    y_true_big = rng.integers(0, 2, size=500)
    y_score_big = y_true_big + rng.normal(0, 0.5, size=500)
    _, lo_b, hi_b = bootstrap_ci(
        y_true_big, y_score_big, compute_auroc,
        n_resamples=1000, seed=4,
    )
    width_big = hi_b - lo_b

    assert width_big < width_small


# ──────────────────────────────────────────────────────────────────────
# paired_bootstrap_diff
# ──────────────────────────────────────────────────────────────────────


def test_paired_bootstrap_diff_zero_when_identical():
    """If both predictors are identical, the diff CI should hug zero."""
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, size=200)
    y_score = y_true + rng.normal(0, 0.4, size=200)
    diff_point, low, high = paired_bootstrap_diff(
        y_true, y_score, y_score, compute_auroc,
        n_resamples=1000, seed=5,
    )
    assert diff_point == 0.0
    assert low == 0.0
    assert high == 0.0


def test_paired_bootstrap_diff_detects_better_predictor():
    """A clearly stronger predictor should give a positive diff."""
    rng = np.random.default_rng(6)
    y_true = rng.integers(0, 2, size=300)
    # A: nearly perfect, B: pure noise
    y_score_a = y_true + rng.normal(0, 0.1, size=300)
    y_score_b = rng.uniform(size=300)
    diff_point, low, high = paired_bootstrap_diff(
        y_true, y_score_a, y_score_b, compute_auroc,
        n_resamples=1000, seed=6,
    )
    assert diff_point > 0
    assert low > 0  # CI excludes zero -> statistically significant


def test_paired_bootstrap_diff_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(
            np.array([0, 1]), np.array([0.5]), np.array([0.5, 0.6]),
            compute_auroc, n_resamples=10,
        )