"""
Classification head for supervised defect classification.
"""

from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    """GAP -> Dropout -> Linear classification head."""

    def __init__(
        self,
        in_features: int = 2048,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features):
        # features: (B, C, H, W) -> logits: (B, 2)
        x = self.pool(features).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

