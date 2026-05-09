import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

class DriftMonitor:
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        """
        reference_data: numpy array of historical/training pixel intensities.
                        Shape can be flattened or (N, C, H, W).
        threshold: p-value threshold for KS test to alert for drift.
        """
        self.reference_data = reference_data.flatten()
        self.threshold = threshold

    def check_drift(self, incoming_data: np.ndarray) -> Tuple[bool, float, float]:
        """
        Performs KS test on input pixel statistics.
        Returns:
            is_drifting (bool)
            ks_statistic (float)
            p_value (float)
        """
        incoming_flat = incoming_data.flatten()
        
        # Subsample if data is too large for fast KS test
        if len(self.reference_data) > 10000:
            ref_sample = np.random.choice(self.reference_data, 10000, replace=False)
        else:
            ref_sample = self.reference_data
            
        if len(incoming_flat) > 10000:
            inc_sample = np.random.choice(incoming_flat, 10000, replace=False)
        else:
            inc_sample = incoming_flat
            
        ks_stat, p_value = ks_2samp(ref_sample, inc_sample)
        
        is_drifting = p_value < self.threshold
        if is_drifting:
            self.alert(ks_stat, p_value)
            
        return is_drifting, float(ks_stat), float(p_value)

    def alert(self, ks_stat: float, p_value: float):
        logger.warning(
            f"DATA DRIFT DETECTED! KS Statistic: {ks_stat:.4f}, p-value: {p_value:.4e} "
            f"(Threshold: {self.threshold})"
        )
