"""Supervised model components for industrial defect detection."""

from src.models.classification_head import ClassificationHead
from src.models.encoder import ResNet50Encoder
from src.models.lit_module import DefectClassifier
from src.models.losses import FocalLoss

__all__ = [
    "ResNet50Encoder",
    "ClassificationHead",
    "FocalLoss",
    "DefectClassifier",
]

try:  # pragma: no cover - optional dependency import
    from src.models.gradcam import get_gradcam, overlay_cam  # noqa: F401

    __all__.extend(["get_gradcam", "overlay_cam"])
except Exception:  # pragma: no cover
    pass
