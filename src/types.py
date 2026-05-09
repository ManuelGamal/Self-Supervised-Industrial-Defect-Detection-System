"""
Shared type definitions for the defect detection project.
"""

from dataclasses import dataclass
from typing import Tuple

from torch import Tensor


@dataclass
class Sample:
    """A single dataset sample."""
    image: Tensor       # (3, 224, 224) float32, ImageNet-normalized
    label: int          # 0=normal, 1=anomalous
    category: str
    path: str


@dataclass
class Prediction:
    """Model prediction output."""
    cls_logits: Tensor  # (B, 2)
    score: Tensor       # (B,) anomaly probability in [0, 1]


@dataclass
class Metrics:
    """Evaluation metrics for a single category."""
    image_auroc: float
    image_aupr: float
    f1: float
    accuracy: float
    category: str
    label_ratio: float


@dataclass
class MetricsWithCI:
    """Evaluation metrics with 95% bootstrap confidence intervals.

    Each metric is paired with a (lower, upper) tuple at 95% confidence.
    Used by the evaluator and consumed by the deployment API.
    """
    image_auroc: float
    image_auroc_ci: Tuple[float, float]
    image_aupr: float
    image_aupr_ci: Tuple[float, float]
    f1: float
    f1_ci: Tuple[float, float]
    accuracy: float
    accuracy_ci: Tuple[float, float]
    threshold: float                       # F1-optimal threshold
    category: str
    fold: int
    n_samples: int