"""
PyTorch Lightning DataModule for MVTec AD.

Wraps :class:`~src.data.mvtec_dataset.MVTecDataset` and the augmentation
pipelines into a single, plug-and-play DataModule for training loops.

Usage::

    dm = MVTecDataModule(
        root="path/to/mvtec-ad",
        category="bottle",
        ratio=100,
        batch_size=32,
    )
    dm.setup()
    for batch in dm.train_dataloader():
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.augmentations import get_eval_transform, get_train_transform
from src.data.mvtec_dataset import MVTecDataset
from src.types import Sample


def _sample_collate_fn(batch: List[Sample]) -> Sample:
    """Collate a list of :class:`Sample` into a single batched Sample."""
    return Sample(
        image=torch.stack([s.image for s in batch]),
        label=[s.label for s in batch],
        category=[s.category for s in batch],
        path=[s.path for s in batch],
    )


class MVTecDataModule(pl.LightningDataModule):
    """Lightning DataModule for a single MVTec AD category.

    Parameters
    ----------
    root : str | Path
        Root directory of the MVTec AD images.
    category : str
        Category name (e.g. ``"bottle"``).
    ratio : int
        Training subset ratio to use (10, 50, or 100).
    split_csv : str | Path, optional
        Explicit split CSV path. If provided, this takes precedence over
        ``ratio`` + ``splits_dir`` naming.
    splits_dir : str | Path
        Directory where split CSVs are stored. Defaults to ``data/splits``.
    batch_size : int
        Batch size for all dataloaders.
    num_workers : int
        Number of parallel data-loading workers.
    image_size : int
        Target spatial size (height = width).
    train_transform : callable, optional
        Override the default training transform.
    eval_transform : callable, optional
        Override the default eval/val/test transform.
    """

    def __init__(
        self,
        root: str | Path = "data/raw",
        category: str = "bottle",
        ratio: int = 100,
        split_csv: Optional[str | Path] = None,
        splits_dir: str | Path = "data/splits",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.category = category
        self.ratio = ratio
        self.split_csv = (
            Path(split_csv)
            if split_csv is not None
            else Path(splits_dir) / f"{category}_{ratio}pct_seed42.csv"
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_transform = train_transform or get_train_transform(image_size)
        self.eval_transform = eval_transform or get_eval_transform(image_size)

        # Populated in setup()
        self.train_dataset: Optional[MVTecDataset] = None
        self.val_dataset: Optional[MVTecDataset] = None
        self.test_dataset: Optional[MVTecDataset] = None

    # ── Lightning hooks ───────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: ARG002
        """Instantiate dataset objects for each split."""
        self.train_dataset = MVTecDataset(
            root=self.root,
            category=self.category,
            split_csv=self.split_csv,
            transform=self.train_transform,
            split_filter="train",
        )
        self.val_dataset = MVTecDataset(
            root=self.root,
            category=self.category,
            split_csv=self.split_csv,
            transform=self.eval_transform,
            split_filter="val",
        )
        self.test_dataset = MVTecDataset(
            root=self.root,
            category=self.category,
            split_csv=self.split_csv,
            transform=self.eval_transform,
            split_filter="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_sample_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_sample_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_sample_collate_fn,
        )
