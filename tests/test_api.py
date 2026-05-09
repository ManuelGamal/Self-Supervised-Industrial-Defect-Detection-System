"""Integration tests for the FastAPI inference service."""

import io
from unittest.mock import patch
from fastapi.testclient import TestClient

# Mock the ONNX session before importing app
with patch("onnxruntime.InferenceSession"):
    from src.deployment.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_stats_empty():
    response = client.get("/stats")
    assert response.status_code == 200


def create_dummy_image_bytes():
    """Create a minimal valid PNG in memory."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_predict_no_model():
    """Should return 503 when model is not loaded."""
    image_bytes = create_dummy_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 503
