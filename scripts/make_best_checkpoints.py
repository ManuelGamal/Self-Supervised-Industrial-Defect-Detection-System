"""
Generate ``results/best_checkpoints.json`` from ``results/results.parquet``.

Stopgap for Engineer 4's deliverable: for each category, pick the fold
with the highest val_auroc and record its checkpoint path on the Kaggle
dataset. Engineer 6 consumes this file to know which 6 of the 18
checkpoints to deploy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# Kaggle dataset layout:
#   checkpoints/supervised/{category}/fold{fold}/best.ckpt
KAGGLE_CHECKPOINT_TEMPLATE = "checkpoints/supervised/{category}/fold{fold}/best.ckpt"

INPUT_PATH = Path("results/results.parquet")
OUTPUT_PATH = Path("results/best_checkpoints.json")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run the W&B aggregate step first."
        )

    df = pd.read_parquet(INPUT_PATH)
    required_cols = {"category", "fold", "auroc", "aupr", "f1", "run_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"results.parquet is missing columns: {missing}")

    # For each category, pick the row with the highest val AUROC.
    best_idx = df.groupby("category")["auroc"].idxmax()
    best_rows = df.loc[best_idx].sort_values("category").reset_index(drop=True)

    best_checkpoints: dict[str, dict] = {}
    for _, row in best_rows.iterrows():
        category = str(row["category"])
        fold = int(row["fold"])
        best_checkpoints[category] = {
            "best_fold": fold,
            "val_auroc": float(row["auroc"]),
            "val_aupr": float(row["aupr"]),
            "val_f1": float(row["f1"]),
            "checkpoint_path": KAGGLE_CHECKPOINT_TEMPLATE.format(
                category=category, fold=fold
            ),
            "wandb_run_id": str(row["run_id"]),
            "run_name": str(row["run_name"]),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(best_checkpoints, f, indent=2)

    # Pretty-print summary
    print(f"[OK] Wrote {OUTPUT_PATH}")
    print(f"\nBest checkpoint per category (by val AUROC):\n")
    print(f"  {'category':<10} {'fold':>4}  {'auroc':>7}  {'aupr':>7}  {'f1':>7}")
    print(f"  {'-'*10} {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}")
    for category, info in best_checkpoints.items():
        print(
            f"  {category:<10} {info['best_fold']:>4}  "
            f"{info['val_auroc']:>7.4f}  "
            f"{info['val_aupr']:>7.4f}  "
            f"{info['val_f1']:>7.4f}"
        )


if __name__ == "__main__":
    main()