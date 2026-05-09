"""
Qualitative analysis: Grad-CAM galleries and failure-case galleries.

For each category, generates two figures:
1. Grad-CAM gallery — N normal + N anomalous samples with attention overlays
2. Failure-case gallery — the 20 samples the model got most wrong

These figures feed the qualitative analysis section of REPORT.md
and are essential for the mandatory negative-results discussion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.gradcam import get_gradcam, overlay_cam
from src.types import Sample


# ImageNet normalization constants — must match the training transform
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a single (3, H, W) ImageNet-normalized tensor back to RGB [0, 1]."""
    image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image_np = image_np * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(image_np, 0.0, 1.0)


def _collect_samples(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> list[dict]:
    """Run inference and collect per-sample records.

    Returns a list of dicts with keys:
        image_tensor (Tensor, (3, H, W))
        label (int)
        score (float)
        path (str)
    """
    model.eval()
    model.to(device)
    records: list[dict] = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, Sample):
                images = batch.image
                labels = batch.label
                paths = batch.path
            elif isinstance(batch, dict):
                images = batch["image"]
                labels = batch["label"]
                paths = batch.get("path", [""] * len(images))
            else:
                images, labels = batch[0], batch[1]
                paths = [""] * len(images)

            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            if torch.is_tensor(labels):
                labels = labels.cpu().numpy().tolist()

            for i, (lbl, prob) in enumerate(zip(labels, probs)):
                records.append({
                    "image_tensor": images[i].cpu(),
                    "label": int(lbl),
                    "score": float(prob),
                    "path": str(paths[i]) if i < len(paths) else "",
                })

    return records


# ──────────────────────────────────────────────────────────────────────
# Grad-CAM gallery
# ──────────────────────────────────────────────────────────────────────


def gradcam_gallery(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_path: Path,
    n_per_class: int = 5,
    device: str = "cpu",
    title: str = "",
) -> None:
    """Render a 2-row × n_per_class-column gallery.

    Top row: normal samples; bottom row: anomalous samples.
    Each cell shows the original image with a Grad-CAM heatmap overlay
    and the predicted score in the title.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _collect_samples(model, dataloader, device=device)
    normals = [r for r in records if r["label"] == 0][:n_per_class]
    anomalous = [r for r in records if r["label"] == 1][:n_per_class]

    if not normals or not anomalous:
        print(f"[WARN] Not enough samples for gallery (normals={len(normals)}, "
              f"anomalous={len(anomalous)}). Skipping.")
        return

    fig, axes = plt.subplots(
        2, n_per_class,
        figsize=(2.5 * n_per_class, 5.5),
    )
    if title:
        fig.suptitle(title, fontsize=12)

    # Make sure axes is always 2D
    if n_per_class == 1:
        axes = axes.reshape(2, 1)

    rows = [("Normal", normals), ("Anomalous", anomalous)]
    for row_idx, (row_label, row_records) in enumerate(rows):
        for col_idx in range(n_per_class):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(row_records):
                ax.axis("off")
                continue

            rec = row_records[col_idx]
            image_tensor = rec["image_tensor"].unsqueeze(0).to(device)
            heatmap = get_gradcam(model, image_tensor)
            image_np = _denormalize(rec["image_tensor"])
            overlay = overlay_cam(image_np, heatmap) / 255.0

            ax.imshow(overlay)
            ax.set_title(f"{row_label}\nscore={rec['score']:.2f}", fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Failure-case gallery
# ──────────────────────────────────────────────────────────────────────


def failure_case_gallery(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_path: Path,
    n_worst: int = 20,
    device: str = "cpu",
    threshold: Optional[float] = None,
    title: str = "",
) -> None:
    """Render the N samples the model got most wrong.

    Wrongness is measured by ``|score - label|`` — i.e. how far the
    predicted probability is from the ground-truth class. Each cell
    shows the image, the true label, predicted score, and a Grad-CAM
    overlay so the analyst can see what the model focused on.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = _collect_samples(model, dataloader, device=device)
    if not records:
        print("[WARN] No samples collected — skipping failure gallery.")
        return

    for rec in records:
        rec["error"] = abs(rec["score"] - rec["label"])

    records.sort(key=lambda r: r["error"], reverse=True)
    worst = records[:n_worst]

    n_cols = 5
    n_rows = (len(worst) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.7 * n_rows),
    )
    if title:
        fig.suptitle(title, fontsize=12)

    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, rec in enumerate(worst):
        ax = axes_flat[i]
        image_tensor = rec["image_tensor"].unsqueeze(0).to(device)
        heatmap = get_gradcam(model, image_tensor)
        image_np = _denormalize(rec["image_tensor"])
        overlay = overlay_cam(image_np, heatmap) / 255.0

        ax.imshow(overlay)
        true_label = "Anomalous" if rec["label"] == 1 else "Normal"
        ax.set_title(
            f"True: {true_label}\nscore={rec['score']:.2f}",
            fontsize=8,
        )
        ax.axis("off")

    # Turn off any unused cells
    for j in range(len(worst), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Convenience: run both galleries for one (category, fold)
# ──────────────────────────────────────────────────────────────────────


def run_qualitative_for_category(
    model: torch.nn.Module,
    dataloader: DataLoader,
    category: str,
    output_dir: Path,
    device: str = "cpu",
    n_per_class: int = 5,
    n_worst: int = 20,
) -> Tuple[Path, Path]:
    """Generate both galleries and return the two output paths."""
    output_dir = Path(output_dir) / category
    gradcam_path = output_dir / "gradcam_gallery.png"
    failures_path = output_dir / "failure_cases.png"

    gradcam_gallery(
        model, dataloader, gradcam_path,
        n_per_class=n_per_class,
        device=device,
        title=f"{category} — Grad-CAM gallery",
    )
    failure_case_gallery(
        model, dataloader, failures_path,
        n_worst=n_worst,
        device=device,
        title=f"{category} — Worst {n_worst} failures",
    )

    return gradcam_path, failures_path