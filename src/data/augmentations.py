"""
Data augmentation / transform pipelines for MVTec AD.

Three presets are provided:

* **train** — lightweight augmentations for supervised fine-tuning.
* **ssl**   — heavier augmentations for contrastive self-supervised pretraining.
* **eval**  — deterministic resize + ImageNet normalisation (no randomness).
"""

from __future__ import annotations

from typing import Callable

import torchvision.transforms as T


# ── ImageNet statistics ───────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> T.Compose:
    """Augmentation pipeline for supervised fine-tuning.

    Applies horizontal flip, brightness/contrast jitter, and Gaussian noise
    (approximated via Gaussian blur with a tiny kernel).
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_ssl_transform(image_size: int = 224, strength: float = 0.5) -> T.Compose:
    """Strong augmentation pipeline for contrastive SSL pretraining.

    Returns a *single view*.  For frameworks that need two views (SimCLR, BYOL),
    call this function twice or use :class:`DualViewTransform`.
    """
    color_jitter = T.ColorJitter(
        brightness=0.8 * strength,
        contrast=0.8 * strength,
        saturation=0.8 * strength,
        hue=0.2 * strength,
    )
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(
                kernel_size=int(0.1 * image_size) | 1,  # ensure odd
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform(image_size: int = 224) -> T.Compose:
    """Deterministic evaluation transform — no augmentation."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class DualViewTransform:
    """Wrapper that returns two independently augmented views of the same image.

    Useful for contrastive SSL methods (SimCLR, BYOL) that require paired views.

    Parameters
    ----------
    base_transform : callable
        A single-view transform (e.g. from :func:`get_ssl_transform`).
    """

    def __init__(self, base_transform: Callable | None = None) -> None:
        self.transform = base_transform or get_ssl_transform()

    def __call__(self, x):  # noqa: ANN001
        return self.transform(x), self.transform(x)
