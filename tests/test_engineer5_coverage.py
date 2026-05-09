import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

import sys
from unittest.mock import MagicMock
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.transforms.functional'] = MagicMock()
sys.modules['src.models.lit_module'] = MagicMock()
sys.modules['timm'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['lightning'] = MagicMock()
sys.modules['lightning.pytorch'] = MagicMock()
sys.modules['pytorch_lightning'] = MagicMock()
sys.modules['pytorch_grad_cam'] = MagicMock()
sys.modules['pytorch_grad_cam.utils'] = MagicMock()
sys.modules['pytorch_grad_cam.utils.image'] = MagicMock()
sys.modules['src.models.gradcam'] = MagicMock()

from src.evaluation.evaluator import evaluate_checkpoint, save_diagnostic_plots, run_inference
from src.evaluation.qualitative import gradcam_gallery, failure_case_gallery, _denormalize, _collect_samples
from src.evaluation.metrics import evaluate_detector, compute_f1_optimal, compute_pixel_iou
import torch

@patch("src.evaluation.evaluator.DefectClassifier")
def test_evaluate_checkpoint(mock_classifier, tmp_path):
    # Mock run_inference to return dummy data
    with patch("src.evaluation.evaluator.run_inference", return_value=(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))):
        res = evaluate_checkpoint("dummy.ckpt", [], "bottle", 1, output_dir=tmp_path, n_bootstrap=10)
        assert res.image_auroc == 1.0

def test_run_inference():
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    loader = [({"image": torch.tensor([1, 2]), "label": torch.tensor([1, 0])})]
    y_true, y_score = run_inference(mock_model, loader)
    assert len(y_true) == 2

def test_save_diagnostic_plots(tmp_path):
    y_true = np.array([0, 1])
    y_score = np.array([0.2, 0.8])
    save_diagnostic_plots(y_true, y_score, 0.5, tmp_path)
    assert (tmp_path / "roc_curve.png").exists()

def test_qualitative_galleries(tmp_path):
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.1, 0.9]])
    
    loader = [({"image": MagicMock(), "label": torch.tensor([1]), "path": [""]})]
    
    with patch("src.evaluation.qualitative.get_gradcam", return_value=np.zeros((10,10))):
        with patch("src.evaluation.qualitative.overlay_cam", return_value=np.zeros((10,10,3))):
            gradcam_gallery(mock_model, loader, tmp_path / "1.png", n_per_class=1)
            failure_case_gallery(mock_model, loader, tmp_path / "2.png", n_worst=1)

def test_metrics_extra_coverage():
    y_true = np.array([0, 1])
    y_score = np.array([0.1, 0.9])
    res = evaluate_detector(y_true, y_score, pred_masks=[np.array([[1]])], gt_masks=[np.array([[1]])])
    assert res["pixel_iou"] == 1.0
    
    from src.evaluation.metrics import print_results
    print_results(res)

    with pytest.raises(ValueError):
        compute_f1_optimal(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        compute_f1_optimal(np.array([0, 0]), np.array([0.1, 0.2]))

    # pixel iou union 0
    assert compute_pixel_iou(np.array([[0]]), np.array([[0]])) == 1.0
