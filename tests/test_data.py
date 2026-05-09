"""Unit tests for the src.data package.

Covers:
  - MVTecDataset  (mvtec_dataset.py)
  - generate_splits  (splits.py)
  - augmentations  (augmentations.py)
  - MVTecDataModule  (datamodule.py)
"""

from pathlib import Path

import torch
import pytest

from src.types import Sample
from src.data.mvtec_dataset import MVTecDataset
from src.data.splits import generate_splits, _collect_samples
from src.data.augmentations import (
    get_train_transform,
    get_ssl_transform,
    get_eval_transform,
    DualViewTransform,
)
from src.data.datamodule import MVTecDataModule

# ── Paths ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _PROJECT_ROOT.parent / "mvtec-ad"
_SPLITS_DIR = _PROJECT_ROOT / "data" / "splits"
_CATEGORY = "bottle"


# ═══════════════════════════════════════════════════════════════════════════
#  MVTecDataset tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def split_csv() -> Path:
    """Return path to the bottle 100pct split CSV."""
    return _SPLITS_DIR / "bottle_100pct_seed42.csv"


@pytest.fixture
def dataset(split_csv: Path) -> MVTecDataset:
    """Return a dataset instance for the bottle category (all splits)."""
    return MVTecDataset(
        root=str(_DATA_ROOT),
        category=_CATEGORY,
        split_csv=str(split_csv),
    )


def test_dataset_returns_sample(dataset: MVTecDataset) -> None:
    """ds[0] returns a valid Sample matching the contract."""
    assert len(dataset) > 0, "Dataset must contain at least one sample"

    sample = dataset[0]
    assert isinstance(sample, Sample), f"Expected Sample, got {type(sample)}"

    # image: Tensor (3, 224, 224), float32
    assert isinstance(sample.image, torch.Tensor)
    assert sample.image.shape == (3, 224, 224)
    assert sample.image.dtype == torch.float32
    assert sample.image.mean().abs() < 2.0, "Image does not look normalized"

    # label: int, 0 or 1
    assert isinstance(sample.label, int)
    assert sample.label in (0, 1)

    # category: str
    assert isinstance(sample.category, str)
    assert sample.category == _CATEGORY

    # path: str pointing to an existing file
    assert isinstance(sample.path, str)
    assert Path(sample.path).exists()


def test_dataset_split_filter(split_csv: Path) -> None:
    """split_filter correctly selects a subset of rows."""
    ds_all = MVTecDataset(root=str(_DATA_ROOT), category=_CATEGORY, split_csv=str(split_csv))
    ds_train = MVTecDataset(root=str(_DATA_ROOT), category=_CATEGORY, split_csv=str(split_csv), split_filter="train")
    ds_val = MVTecDataset(root=str(_DATA_ROOT), category=_CATEGORY, split_csv=str(split_csv), split_filter="val")
    ds_test = MVTecDataset(root=str(_DATA_ROOT), category=_CATEGORY, split_csv=str(split_csv), split_filter="test")

    assert len(ds_train) > 0
    assert len(ds_val) > 0
    assert len(ds_test) > 0
    assert len(ds_train) + len(ds_val) + len(ds_test) == len(ds_all)


# ═══════════════════════════════════════════════════════════════════════════
#  splits.py tests
# ═══════════════════════════════════════════════════════════════════════════

def test_collect_samples() -> None:
    """_collect_samples finds images for the bottle category."""
    samples = _collect_samples(_DATA_ROOT, "bottle")
    assert len(samples) > 0
    # Each sample is (relative_path, label)
    path, label = samples[0]
    assert isinstance(path, str)
    assert label in (0, 1)


def test_generate_splits_count(tmp_path: Path) -> None:
    """generate_splits creates 18 CSVs (6 categories x 3 ratios)."""
    paths = generate_splits(root=_DATA_ROOT, out_dir=tmp_path)
    assert len(paths) == 18
    for p in paths:
        assert p.exists()
        assert p.suffix == ".csv"


def test_split_no_leakage(tmp_path: Path) -> None:
    """No image path appears in more than one split (train/val/test)."""
    import csv

    generate_splits(
        root=_DATA_ROOT,
        out_dir=tmp_path,
        categories=["bottle"],
    )
    csv_path = tmp_path / "bottle_100pct_seed42.csv"

    split_sets: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split_sets[row["split"]].add(row["image_path"])

    # No overlap between any two splits
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])

    # All splits non-empty
    assert len(split_sets["train"]) > 0
    assert len(split_sets["val"]) > 0
    assert len(split_sets["test"]) > 0


def test_split_ratios_approximate(tmp_path: Path) -> None:
    """Split sizes roughly match the requested ratios (within 10% tolerance)."""
    import csv

    generate_splits(
        root=_DATA_ROOT,
        out_dir=tmp_path,
        categories=["bottle"],
    )
    csv_path = tmp_path / "bottle_100pct_seed42.csv"

    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts[row["split"]] += 1

    total = sum(counts.values())
    assert 0.60 <= counts["train"] / total <= 0.80  # ~70%
    assert 0.05 <= counts["val"] / total <= 0.25     # ~15%
    assert 0.05 <= counts["test"] / total <= 0.25    # ~15%


# ═══════════════════════════════════════════════════════════════════════════
#  augmentations.py tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def dummy_pil_image():
    """Create a dummy 256x256 RGB PIL image."""
    from PIL import Image
    import numpy as np
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_train_transform_output(dummy_pil_image) -> None:
    """Train transform returns (3, 224, 224) float32 tensor."""
    t = get_train_transform(224)
    out = t(dummy_pil_image)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_ssl_transform_output(dummy_pil_image) -> None:
    """SSL transform returns (3, 224, 224) float32 tensor."""
    t = get_ssl_transform(224)
    out = t(dummy_pil_image)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_eval_transform_output(dummy_pil_image) -> None:
    """Eval transform returns (3, 224, 224) float32 tensor."""
    t = get_eval_transform(224)
    out = t(dummy_pil_image)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_dual_view_transform(dummy_pil_image) -> None:
    """DualViewTransform returns two distinct tensors."""
    dvt = DualViewTransform()
    v1, v2 = dvt(dummy_pil_image)
    assert isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor)
    assert v1.shape == v2.shape == (3, 224, 224)
    # Two random views should almost never be identical
    assert not torch.equal(v1, v2)


# ═══════════════════════════════════════════════════════════════════════════
#  datamodule.py tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def datamodule() -> MVTecDataModule:
    """Return a DataModule for bottle with small batch size."""
    dm = MVTecDataModule(
        root=str(_DATA_ROOT),
        category=_CATEGORY,
        ratio=100,
        splits_dir=str(_SPLITS_DIR),
        batch_size=4,
        num_workers=0,  # avoid multiprocessing in tests
    )
    dm.setup()
    return dm


def test_datamodule_setup(datamodule: MVTecDataModule) -> None:
    """DataModule creates non-empty train/val/test datasets."""
    assert len(datamodule.train_dataset) > 0
    assert len(datamodule.val_dataset) > 0
    assert len(datamodule.test_dataset) > 0


def test_datamodule_train_dataloader(datamodule: MVTecDataModule) -> None:
    """train_dataloader yields batches with correct shape."""
    batch = next(iter(datamodule.train_dataloader()))
    # Sample is a dataclass, DataLoader collates field-by-field
    assert batch.image.shape[0] <= 4
    assert batch.image.shape[1:] == (3, 224, 224)


def test_datamodule_val_dataloader(datamodule: MVTecDataModule) -> None:
    """val_dataloader yields batches."""
    batch = next(iter(datamodule.val_dataloader()))
    assert batch.image.shape[1:] == (3, 224, 224)


def test_datamodule_test_dataloader(datamodule: MVTecDataModule) -> None:
    """test_dataloader yields batches."""
    batch = next(iter(datamodule.test_dataloader()))
    assert batch.image.shape[1:] == (3, 224, 224)
