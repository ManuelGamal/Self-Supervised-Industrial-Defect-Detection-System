"""
PyTorch Lightning module for supervised industrial defect classification.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
)

from src.models.classification_head import ClassificationHead
from src.models.encoder import ResNet50Encoder
from src.models.losses import FocalLoss
from src.types import Sample


class DefectClassifier(pl.LightningModule):
    """Lightning module wrapping encoder, head, and supervised metrics."""

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        warmup_epochs: int = 2,  # Reserved for trainer-level schedule upgrades
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ResNet50Encoder(pretrained=pretrained)
        self.head = ClassificationHead(
            in_features=self.encoder.feature_dim,
            num_classes=2,
            dropout=dropout,
        )
        self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

        self.val_auroc = BinaryAUROC()
        self.val_f1 = BinaryF1Score()
        self.val_aupr = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()
        self.test_f1 = BinaryF1Score()
        self.test_aupr = BinaryAveragePrecision()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.head(features)
        return logits

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Support dict, Sample dataclass, and tuple/list batches."""
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch["label"]
        elif isinstance(batch, Sample):
            x = batch.image
            y = batch.label
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        if not torch.is_tensor(y):
            y = torch.as_tensor(y, device=x.device)
        y = y.long()
        return x, y

    def _step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self._unpack_batch(batch)
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]
        return loss, probs, y

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        loss, _, _ = self._step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        loss, probs, y = self._step(batch)
        self.val_auroc.update(probs, y)
        self.val_f1.update(probs, y)
        self.val_aupr.update(probs, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val/auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute())
        self.log("val/aupr", self.val_aupr.compute())
        self.val_auroc.reset()
        self.val_f1.reset()
        self.val_aupr.reset()

    def test_step(self, batch: Any, batch_idx: int) -> None:  # noqa: ARG002
        _, probs, y = self._step(batch)
        self.test_auroc.update(probs, y)
        self.test_f1.update(probs, y)
        self.test_aupr.update(probs, y)

    def on_test_epoch_end(self) -> None:
        self.log("test/auroc", self.test_auroc.compute())
        self.log("test/f1", self.test_f1.compute())
        self.log("test/aupr", self.test_aupr.compute())
        self.test_auroc.reset()
        self.test_f1.reset()
        self.test_aupr.reset()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return [optimizer], [scheduler]

