"""
Encoder module for supervised defect classification.
"""

from __future__ import annotations

import warnings

import timm
import torch.nn as nn


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature encoder from timm."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            self.backbone = timm.create_model(
                "resnet50",
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
            )
        except Exception as exc:  # pragma: no cover - fallback for offline envs
            if not pretrained:
                raise
            warnings.warn(
                f"Failed to load pretrained timm weights ({exc}); "
                "falling back to randomly initialized weights.",
                stacklevel=2,
            )
            self.backbone = timm.create_model(
                "resnet50",
                pretrained=False,
                num_classes=0,
                global_pool="",
            )
        self.feature_dim = 2048

    def forward(self, x):
        # x: (B, 3, 224, 224) -> features: (B, 2048, 7, 7)
        return self.backbone(x)

