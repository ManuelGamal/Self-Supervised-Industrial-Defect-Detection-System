"""
PyTorch Dataset for MVTec AD, driven by a split CSV file.

Each row in the CSV must have columns:
    image_path  – path relative to ``root`` (e.g. bottle/train/good/000.png)
    label       – 0 (normal) or 1 (anomalous)
    split       – one of ``train``, ``val``, ``test``
    category    – MVTec AD category name
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.types import Sample


# ── Default transform (ImageNet-normalized, 224×224) ──────────────────────
_DEFAULT_TRANSFORM: T.Compose = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MVTecDataset(Dataset):
    """CSV-driven PyTorch ``Dataset`` for MVTec AD.

    Parameters
    ----------
    root : str | Path
        Root directory that contains the actual image files.
        ``image_path`` entries in the CSV are resolved relative to this path.
    category : str
        MVTec AD category name (e.g. ``"bottle"``).  Only rows whose
        ``category`` column matches are kept.
    split_csv : str | Path
        Path to the split CSV file.
    transform : callable, optional
        Torchvision-style transform applied to each PIL image.
        Defaults to resize-224 → ToTensor → ImageNet-normalize.
    split_filter : str, optional
        If provided, keep only rows whose ``split`` column matches
        (e.g. ``"train"``, ``"val"``, ``"test"``).  If *None*, all rows
        for the given category are loaded.
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split_csv: str | Path,
        transform: Optional[Callable] = None,
        split_filter: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.transform = transform or _DEFAULT_TRANSFORM

        # Parse the CSV and keep only rows matching the requested category
        # (and optionally the requested split).
        self._samples: list[dict] = []
        with open(split_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["category"] != category:
                    continue
                if split_filter and row.get("split") != split_filter:
                    continue
                self._samples.append(
                    {
                        "image_path": row["image_path"],
                        "label": int(row["label"]),
                        "category": row["category"],
                    }
                )

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        entry = self._samples[idx]

        img_path = self.root / entry["image_path"]
        image = Image.open(img_path).convert("RGB")
        image_tensor: torch.Tensor = self.transform(image)

        return Sample(
            image=image_tensor,
            label=entry["label"],
            category=entry["category"],
            path=str(img_path),
        )
