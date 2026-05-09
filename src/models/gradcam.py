"""
Grad-CAM utilities for supervised defect classifier interpretability.
"""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer=None,
) -> np.ndarray:
    """
    Return Grad-CAM heatmap for a single input tensor.

    Parameters
    ----------
    model:
        Model expected to output classification logits.
    image_tensor:
        Tensor of shape (1, 3, 224, 224).
    target_layer:
        Layer to target; defaults to last block in encoder layer4.
    """
    model.eval()
    if target_layer is None:
        target_layer = model.encoder.backbone.layer4[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor)[0]
    return np.clip(grayscale_cam, 0.0, 1.0)


def overlay_cam(image_np: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend Grad-CAM heatmap onto RGB image in [0, 1]."""
    return show_cam_on_image(image_np, heatmap, use_rgb=True)
