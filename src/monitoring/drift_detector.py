"""
Feature distribution drift detection.
Monitors embedding statistics over time to trigger retraining.
"""

import numpy as np
from collections import deque
from scipy import stats


class DriftDetector:
    """
    Sliding window drift detector using KS test on feature distributions.

    Compares recent embedding statistics against a reference distribution
    (computed from the training set) to flag data drift.
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        window_size: int = 1000,
        significance_level: float = 0.05,
    ):
        self.reference = reference_embeddings
        self.window_size = window_size
        self.significance_level = significance_level
        self.window = deque(maxlen=window_size)
        self.drift_history = []

    def update(self, embedding: np.ndarray):
        """Add a new embedding to the sliding window."""
        self.window.append(embedding.flatten())

    def check_drift(self) -> dict:
        """
        Run KS test comparing window embeddings to reference distribution.

        Returns:
            dict with drift status, p-value, and recommendation
        """
        if len(self.window) < self.window_size // 2:
            return {"drift_detected": False, "reason": "Insufficient data"}

        window_arr = np.array(list(self.window))
        ref_sample = self.reference[
            np.random.choice(len(self.reference), size=len(window_arr), replace=True)
        ]

        # KS test on mean embedding per dimension (fast approximation)
        window_means = window_arr.mean(axis=0)
        ref_means = ref_sample.mean(axis=0)

        _, p_value = stats.ks_2samp(window_means, ref_means)
        drift_detected = p_value < self.significance_level

        result = {
            "drift_detected": drift_detected,
            "p_value": round(float(p_value), 6),
            "window_size": len(self.window),
            "recommendation": "Trigger retraining" if drift_detected else "No action needed",
        }
        self.drift_history.append(result)
        return result

    def should_retrain(self) -> bool:
        """Return True if retraining is recommended."""
        result = self.check_drift()
        return result["drift_detected"]
