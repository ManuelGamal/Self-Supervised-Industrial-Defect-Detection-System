from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from src.evaluation import evaluator as evaluator_module
from src.evaluation import qualitative as qualitative_module
from src.evaluation.evaluator import (
    evaluate_checkpoint,
    run_inference,
    save_diagnostic_plots,
)
from src.evaluation.metrics import (
    compute_f1_optimal,
    compute_pixel_iou,
    evaluate_detector,
    print_results,
)
from src.evaluation.qualitative import (
    _collect_samples,
    _denormalize,
    failure_case_gallery,
    gradcam_gallery,
    run_qualitative_for_category,
)


class TinyDefectModel(torch.nn.Module):
    """Small deterministic model for evaluation tests.

    It returns alternating logits:
    sample 0 -> high anomaly probability
    sample 1 -> low anomaly probability
    sample 2 -> high anomaly probability
    ...
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.tensor(
            [[0.1, 2.0], [2.0, 0.1]],
            dtype=x.dtype,
            device=x.device,
        )
        repeats = (batch_size + 1) // 2
        return logits.repeat(repeats, 1)[:batch_size]


def make_loader() -> list[dict]:
    """Create a tiny dataloader-like list with both classes."""
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([1, 0, 1, 0])
    paths = ["anom_1.png", "normal_1.png", "anom_2.png", "normal_2.png"]

    return [
        {
            "image": images,
            "label": labels,
            "path": paths,
        }
    ]


def test_run_inference_collects_labels_and_scores():
    model = TinyDefectModel()
    loader = make_loader()

    y_true, y_score = run_inference(model, loader, device="cpu")

    assert y_true.shape == (4,)
    assert y_score.shape == (4,)
    assert y_true.tolist() == [1, 0, 1, 0]
    assert np.all((y_score >= 0.0) & (y_score <= 1.0))


def test_save_diagnostic_plots_creates_expected_files(tmp_path: Path):
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    save_diagnostic_plots(
        y_true=y_true,
        y_score=y_score,
        threshold=0.5,
        output_dir=tmp_path,
        title_prefix="test — ",
    )

    assert (tmp_path / "roc_curve.png").exists()
    assert (tmp_path / "pr_curve.png").exists()
    assert (tmp_path / "confusion_matrix.png").exists()


def test_evaluate_checkpoint_saves_metrics_and_plots(monkeypatch, tmp_path: Path):
    model = TinyDefectModel()

    monkeypatch.setattr(
        evaluator_module.DefectClassifier,
        "load_from_checkpoint",
        staticmethod(lambda *args, **kwargs: model),
    )

    metrics = evaluate_checkpoint(
        checkpoint_path="dummy.ckpt",
        dataloader=make_loader(),
        category="bottle",
        fold=1,
        device="cpu",
        output_dir=tmp_path,
        n_bootstrap=20,
        seed=42,
    )

    output_dir = tmp_path / "bottle" / "fold1"

    assert metrics.category == "bottle"
    assert metrics.fold == 1
    assert metrics.n_samples == 4
    assert metrics.image_auroc == pytest.approx(1.0)

    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "roc_curve.png").exists()
    assert (output_dir / "pr_curve.png").exists()
    assert (output_dir / "confusion_matrix.png").exists()

    with open(output_dir / "metrics.json", "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["category"] == "bottle"
    assert saved["fold"] == 1


def test_denormalize_returns_rgb_image_in_valid_range():
    image = torch.zeros(3, 8, 8)

    denorm = _denormalize(image)

    assert denorm.shape == (8, 8, 3)
    assert denorm.min() >= 0.0
    assert denorm.max() <= 1.0


def test_collect_samples_returns_records():
    model = TinyDefectModel()
    records = _collect_samples(model, make_loader(), device="cpu")

    assert len(records) == 4
    assert {"image_tensor", "label", "score", "path"} <= set(records[0].keys())
    assert records[0]["label"] in {0, 1}
    assert 0.0 <= records[0]["score"] <= 1.0


def test_gradcam_gallery_creates_file(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        qualitative_module,
        "get_gradcam",
        lambda model, image_tensor: np.zeros((8, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        qualitative_module,
        "overlay_cam",
        lambda image_np, heatmap: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    output_path = tmp_path / "gradcam_gallery.png"

    gradcam_gallery(
        model=TinyDefectModel(),
        dataloader=make_loader(),
        output_path=output_path,
        n_per_class=1,
        device="cpu",
        title="test gallery",
    )

    assert output_path.exists()


def test_failure_case_gallery_creates_file(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        qualitative_module,
        "get_gradcam",
        lambda model, image_tensor: np.zeros((8, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        qualitative_module,
        "overlay_cam",
        lambda image_np, heatmap: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    output_path = tmp_path / "failure_cases.png"

    failure_case_gallery(
        model=TinyDefectModel(),
        dataloader=make_loader(),
        output_path=output_path,
        n_worst=2,
        device="cpu",
        title="test failures",
    )

    assert output_path.exists()


def test_run_qualitative_for_category_returns_paths(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        qualitative_module,
        "get_gradcam",
        lambda model, image_tensor: np.zeros((8, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        qualitative_module,
        "overlay_cam",
        lambda image_np, heatmap: np.zeros((8, 8, 3), dtype=np.uint8),
    )

    gradcam_path, failures_path = run_qualitative_for_category(
        model=TinyDefectModel(),
        dataloader=make_loader(),
        category="bottle",
        output_dir=tmp_path,
        device="cpu",
        n_per_class=1,
        n_worst=2,
    )

    assert gradcam_path.exists()
    assert failures_path.exists()


def test_metrics_extra_coverage(capsys):
    y_true = np.array([0, 1])
    y_score = np.array([0.1, 0.9])

    results = evaluate_detector(
        y_true,
        y_score,
        pred_masks=[np.array([[1]])],
        gt_masks=[np.array([[1]])],
    )

    assert results["pixel_iou"] == 1.0
    assert results["auroc"] == pytest.approx(1.0)

    print_results(results)
    captured = capsys.readouterr()
    assert "Evaluation Results" in captured.out

    with pytest.raises(ValueError):
        compute_f1_optimal(np.array([]), np.array([]))

    with pytest.raises(ValueError):
        compute_f1_optimal(np.array([0, 0]), np.array([0.1, 0.2]))

    assert compute_pixel_iou(np.array([[0]]), np.array([[0]])) == 1.0