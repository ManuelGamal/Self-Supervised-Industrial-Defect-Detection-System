"""Unit tests for supervised model components."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.models as models_pkg
from src.models.anomaly_detector import AnomalyDetector, UNetDecoder
from src.models.classification_head import ClassificationHead
from src.models.encoder import ResNet50Encoder
from src.models.export_onnx import main as export_main
from src.models.gradcam import get_gradcam, overlay_cam
from src.models.lit_module import DefectClassifier
from src.models.losses import FocalLoss
from src.types import Sample


def test_encoder_output_shape() -> None:
    m = ResNet50Encoder(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    assert m(x).shape == (2, 2048, 7, 7)


def test_head_output_shape() -> None:
    h = ClassificationHead()
    x = torch.randn(2, 2048, 7, 7)
    assert h(x).shape == (2, 2)


def test_focal_loss_positive() -> None:
    loss = FocalLoss()
    logits = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    assert loss(logits, y).item() > 0


def test_focal_loss_near_zero_when_correct() -> None:
    loss = FocalLoss()
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    y = torch.tensor([0, 1])
    assert loss(logits, y).item() < 0.01


def test_lit_module_forward() -> None:
    m = DefectClassifier(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    assert m(x).shape == (2, 2)


def test_lit_module_training_step() -> None:
    m = DefectClassifier(pretrained=False)
    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "label": torch.tensor([0, 1, 0, 1]),
    }
    loss = m.training_step(batch, 0)
    assert loss.requires_grad
    assert loss.item() > 0


def test_models_init_exports() -> None:
    assert hasattr(models_pkg, "ResNet50Encoder")
    assert hasattr(models_pkg, "ClassificationHead")
    assert hasattr(models_pkg, "FocalLoss")
    assert hasattr(models_pkg, "DefectClassifier")
    assert "ResNet50Encoder" in models_pkg.__all__


def test_lit_module_sample_and_tuple_batches() -> None:
    m = DefectClassifier(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    y = [0, 1, 0, 1]

    sample = Sample(image=x, label=y, category=["bottle"] * 4, path=["p"] * 4)
    loss_sample = m.training_step(sample, 0)
    assert loss_sample.item() > 0

    loss_tuple = m.training_step((x, torch.tensor(y)), 0)
    assert loss_tuple.item() > 0


def test_lit_module_val_test_and_optimizers() -> None:
    m = DefectClassifier(pretrained=False)
    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "label": torch.tensor([0, 1, 0, 1]),
    }
    m.validation_step(batch, 0)
    m.on_validation_epoch_end()
    m.test_step(batch, 0)
    m.on_test_epoch_end()

    optimizers, schedulers = m.configure_optimizers()
    assert len(optimizers) == 1
    assert len(schedulers) == 1


def test_anomaly_detector_fit_score_predict() -> None:
    detector = AnomalyDetector(backbone="resnet18", k=1, device="cpu")

    dummy = torch.randn(4, 3, 224, 224)
    loader = [dummy[:2], dummy[2:]]
    detector.fit(loader)

    scores = detector.anomaly_score(loader)
    preds = detector.predict(loader, threshold=0.0)

    assert scores.shape == (4,)
    assert preds.shape == (4,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_unet_decoder_forward() -> None:
    decoder = UNetDecoder()
    skips = [
        torch.randn(2, 64, 112, 112),
        torch.randn(2, 256, 56, 56),
        torch.randn(2, 512, 28, 28),
        torch.randn(2, 1024, 14, 14),
        torch.randn(2, 2048, 7, 7),
    ]
    out = decoder(skips)
    assert out.shape == (2, 1, 224, 224)


def test_anomaly_detector_requires_fit() -> None:
    detector = AnomalyDetector(backbone="resnet18", k=1, device="cpu")
    with pytest.raises(AssertionError):
        _ = detector.anomaly_score([torch.randn(2, 3, 224, 224)])


def test_gradcam_helpers(monkeypatch) -> None:
    class DummyCam:
        def __init__(self, model, target_layers):  # noqa: ANN001
            self.model = model
            self.target_layers = target_layers

        def __call__(self, input_tensor):  # noqa: ANN001
            b = input_tensor.shape[0]
            return np.ones((b, 224, 224), dtype=np.float32) * 0.5

    class DummyShow:
        @staticmethod
        def run(image_np, heatmap, use_rgb):  # noqa: ANN001
            assert use_rgb
            return (image_np * 255).astype(np.uint8)

    monkeypatch.setattr("src.models.gradcam.GradCAM", DummyCam)
    monkeypatch.setattr(
        "src.models.gradcam.show_cam_on_image",
        DummyShow.run,
    )

    model = DefectClassifier(pretrained=False)
    image_tensor = torch.randn(1, 3, 224, 224)
    heatmap = get_gradcam(model, image_tensor)
    assert heatmap.shape == (224, 224)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0

    image_np = np.random.rand(224, 224, 3).astype(np.float32)
    overlay = overlay_cam(image_np, heatmap)
    assert overlay.shape == (224, 224, 3)


def test_export_onnx_main(monkeypatch, tmp_path: Path) -> None:
    ckpt = tmp_path / "smoke.ckpt"
    out = tmp_path / "model.onnx"

    dummy_model = DefectClassifier(pretrained=False)

    class DummySession:
        def __init__(self, *args, **kwargs):  # noqa: ANN003, ANN002
            pass

        def run(self, _unused, feed):  # noqa: ANN001
            x = torch.from_numpy(feed["image"])
            with torch.no_grad():
                logits = dummy_model(x).cpu().numpy()
            return [logits]

    monkeypatch.setattr(
        "src.models.export_onnx.DefectClassifier.load_from_checkpoint",
        lambda *args, **kwargs: dummy_model,
    )
    monkeypatch.setattr(
        "src.models.export_onnx.torch.onnx.export",
        lambda *args, **kwargs: out.write_bytes(b"fake-onnx"),
    )
    monkeypatch.setattr(
        "src.models.export_onnx.ort.InferenceSession",
        DummySession,
    )
    monkeypatch.setattr(
        "src.models.export_onnx.argparse.ArgumentParser.parse_args",
        lambda self: SimpleNamespace(checkpoint=str(ckpt), output=str(out), opset=17),
    )

    export_main()
    assert out.exists()
