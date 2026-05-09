"""
Generate stratified nested train / val / test split CSV files for MVTec AD.

Produces **18 CSV files** (6 categories × 3 nested training subset ratios)
under ``data/splits/``. The ratios represent the percentage of training
data retained: 100%, 50%, and 10%.

The splits are nested: Train(10%) ⊂ Train(50%) ⊂ Train(100%).
The Validation and Test sets remain identical across all ratios for a given
category to ensure fair evaluation.

Usage::

    python -m src.data.splits --root path/to/mvtec-ad --out data/splits
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

CATEGORIES: list[str] = [
    "bottle", "capsule", "carpet", "hazelnut", "leather", "pill",
]

# We will create subsets of the training data
TRAIN_RATIOS = [100, 50, 10]
SEED: int = 42

def _collect_samples(root: Path, category: str) -> List[Tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    cat_dir = root / category

    # Train – only "good" images
    train_good = cat_dir / "train" / "good"
    if train_good.exists():
        for p in sorted(train_good.glob("*.png")):
            rel = p.relative_to(root).as_posix()
            samples.append((rel, 0))

    # Test – good + all defect sub-folders
    test_dir = cat_dir / "test"
    if test_dir.exists():
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            label = 0 if defect_dir.name == "good" else 1
            for p in sorted(defect_dir.glob("*.png")):
                rel = p.relative_to(root).as_posix()
                samples.append((rel, label))

    return samples

def generate_splits(
    root: Path,
    out_dir: Path,
    categories: list[str] | None = None,
    seed: int = SEED,
) -> list[Path]:
    categories = categories or CATEGORIES
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for category in categories:
        samples = _collect_samples(root, category)
        if not samples:
            raise FileNotFoundError(f"No images found for category '{category}' in {root}")

        paths, labels = zip(*samples)
        paths_list = list(paths)
        labels_list = list(labels)

        # 1. First, split into a master Train (70%) and ValTest (30%)
        # Stratify ensures train and valtest have the same anomaly ratio
        train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
            paths_list, labels_list, test_size=0.30, random_state=seed, stratify=labels_list
        )

        # 2. Split ValTest into identical Val (15%) and Test (15%)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            valtest_paths, valtest_labels, test_size=0.50, random_state=seed, stratify=valtest_labels
        )

        # Base 100% training set
        train_subsets = {100: (train_paths, train_labels)}

        # Create nested 50% subset from the 100% set
        train_paths_50, _, train_labels_50, _ = train_test_split(
            train_paths, train_labels, train_size=0.50, random_state=seed, stratify=train_labels
        )
        train_subsets[50] = (train_paths_50, train_labels_50)

        # Create nested 10% subset from the 50% set (which is 20% of the 50% set)
        train_paths_10, _, train_labels_10, _ = train_test_split(
            train_paths_50, train_labels_50, train_size=0.20, random_state=seed, stratify=train_labels_50
        )
        train_subsets[10] = (train_paths_10, train_labels_10)

        # Write the 3 CSV files for this category
        for pct in TRAIN_RATIOS:
            fname = f"{category}_{pct}pct_seed{seed}.csv"
            out_path = out_dir / fname
            
            cur_train_paths, cur_train_labels = train_subsets[pct]
            
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["image_path", "label", "split", "category"])
                for p, label in zip(cur_train_paths, cur_train_labels):
                    writer.writerow([p, label, "train", category])
                for p, label in zip(val_paths, val_labels):
                    writer.writerow([p, label, "val", category])
                for p, label in zip(test_paths, test_labels):
                    writer.writerow([p, label, "test", category])

            written.append(out_path)

    return written

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MVTec AD nested split CSVs")
    parser.add_argument("--root", type=str, required=True, help="Path to mvtec-ad root")
    parser.add_argument("--out", type=str, default="data/splits", help="Output directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    paths = generate_splits(root=Path(args.root), out_dir=Path(args.out), seed=args.seed)
    print(f"[OK] Generated {len(paths)} nested split CSVs:")
    for p in paths:
        print(f"   {p.name}")

if __name__ == "__main__":
    main()
