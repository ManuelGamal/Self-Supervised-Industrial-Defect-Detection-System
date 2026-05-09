"""
End-to-end evaluator for a single trained checkpoint.

Loads a DefectClassifier from a Lightning checkpoint, runs it on a
DataLoader, computes metrics with bootstrap confidence intervals, and
saves diagnostic plots (ROC, PR, confusion matrix) plus a metrics JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.metrics import (
    compute_accuracy,
    compute_aupr,
    compute_auroc,
    compute_f1,
    compute_f1_optimal,
)
from src.models.lit_module import DefectClassifier
from src.types import MetricsWithCI, Sample


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model on the dataloader and collect (y_true, y_score).

    y_score is the softmax probability of class 1 (anomalous).
    """
    model.eval()
    model.to(device)

    all_scores: list[float] = []
    all_labels: list[int] = []

    for batch in dataloader:
        # The MVTec datamodule yields a Sample with batched fields.
        if isinstance(batch, Sample):
            images = batch.image
            labels = batch.label
        elif isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        elif isinstance(batch, (tuple, list)):
            images, labels = batch[0], batch[1]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(anomalous)

        all_scores.extend(probs.cpu().numpy().tolist())
        if torch.is_tensor(labels):
            all_labels.extend(labels.cpu().numpy().tolist())
        else:
            all_labels.extend(list(labels))

    return np.asarray(all_labels, dtype=int), np.asarray(all_scores, dtype=float)


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────


def save_diagnostic_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Save ROC curve, PR curve, and confusion matrix as PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc_value = compute_auroc(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUROC = {auroc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix}ROC Curve".strip())
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr_value = compute_aupr(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"AUPR = {aupr_value:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix}PR Curve".strip())
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix at the given threshold
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Anomalous"])
    ax.set_yticks([0, 1], ["Normal", "Anomalous"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title_prefix}Confusion Matrix (thr={threshold:.3f})".strip())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataloader: DataLoader,
    category: str,
    fold: int,
    device: str = "cpu",
    output_dir: str | Path = "results/eval",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> MetricsWithCI:
    """Load a checkpoint, run evaluation, save artefacts, return MetricsWithCI.

    Saves to ``{output_dir}/{category}/fold{fold}/``:
        - metrics.json
        - roc_curve.png
        - pr_curve.png
        - confusion_matrix.png
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir) / category / f"fold{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = DefectClassifier.load_from_checkpoint(
        checkpoint_path, map_location=device
    )

    # Inference
    y_true, y_score = run_inference(model, dataloader, device=device)

    # Pick the F1-optimal threshold
    f1_value, threshold = compute_f1_optimal(y_true, y_score)

    # Point estimates
    auroc_value = compute_auroc(y_true, y_score)
    aupr_value = compute_aupr(y_true, y_score)
    accuracy_value = compute_accuracy(y_true, y_score, threshold=threshold)

    # Bootstrap CIs
    _, auroc_lo, auroc_hi = bootstrap_ci(
        y_true, y_score, compute_auroc, n_resamples=n_bootstrap, seed=seed
    )
    _, aupr_lo, aupr_hi = bootstrap_ci(
        y_true, y_score, compute_aupr, n_resamples=n_bootstrap, seed=seed
    )

    def f1_at_fixed_threshold(yt, ys):
        return compute_f1(yt, ys, threshold=threshold)

    def accuracy_at_fixed_threshold(yt, ys):
        return compute_accuracy(yt, ys, threshold=threshold)

    _, f1_lo, f1_hi = bootstrap_ci(
        y_true, y_score, f1_at_fixed_threshold,
        n_resamples=n_bootstrap, seed=seed,
    )
    _, acc_lo, acc_hi = bootstrap_ci(
        y_true, y_score, accuracy_at_fixed_threshold,
        n_resamples=n_bootstrap, seed=seed,
    )

    metrics = MetricsWithCI(
        image_auroc=auroc_value,
        image_auroc_ci=(auroc_lo, auroc_hi),
        image_aupr=aupr_value,
        image_aupr_ci=(aupr_lo, aupr_hi),
        f1=f1_value,
        f1_ci=(f1_lo, f1_hi),
        accuracy=accuracy_value,
        accuracy_ci=(acc_lo, acc_hi),
        threshold=float(threshold),
        category=category,
        fold=fold,
        n_samples=int(len(y_true)),
    )

    # Save plots
    save_diagnostic_plots(
        y_true, y_score, threshold,
        output_dir=output_dir,
        title_prefix=f"{category} fold{fold} — ",
    )

    # Save metrics JSON
    metrics_json = {
        "image_auroc": metrics.image_auroc,
        "image_auroc_ci": list(metrics.image_auroc_ci),
        "image_aupr": metrics.image_aupr,
        "image_aupr_ci": list(metrics.image_aupr_ci),
        "f1": metrics.f1,
        "f1_ci": list(metrics.f1_ci),
        "accuracy": metrics.accuracy,
        "accuracy_ci": list(metrics.accuracy_ci),
        "threshold": metrics.threshold,
        "category": metrics.category,
        "fold": metrics.fold,
        "n_samples": metrics.n_samples,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    return metrics