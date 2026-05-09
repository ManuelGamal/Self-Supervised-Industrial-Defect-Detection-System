"""
FastAPI Inference Service for Defect Detection.

Endpoints:
    POST /predict       - Single image defect prediction
    POST /predict/batch - Batch image prediction
    GET  /health        - Health check
    GET  /metrics       - Prometheus metrics
"""

import io
import time
import logging
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as T

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Self-supervised defect detection inference service",
    version="1.0.0",
)

# Global inference session
session: ort.InferenceSession = None
prediction_log: list = []

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(onnx_path: str):
    """Load ONNX model for inference."""
    global session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    logger.info(f"Model loaded from {onnx_path}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to model input tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).numpy()
    return tensor


def run_inference(tensor: np.ndarray) -> dict:
    """Run ONNX inference and return prediction dict."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    score = float(outputs[0][0])
    is_defect = score > 0.5
    return {
        "anomaly_score": round(score, 4),
        "is_defect": is_defect,
        "label": "defect" if is_defect else "normal",
        "confidence": round(score if is_defect else 1 - score, 4),
    }


@app.on_event("startup")
async def startup():
    onnx_path = "checkpoints/model_quantized.onnx"
    if Path(onnx_path).exists():
        load_model(onnx_path)
    else:
        logger.warning(f"Model not found at {onnx_path}. POST endpoints will fail.")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "predictions_served": len(prediction_log),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict defect status for a single image.

    Returns anomaly score, binary prediction, and confidence.
    """
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start_time = time.time()
    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes)
    result = run_inference(tensor)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    result["latency_ms"] = latency_ms
    result["filename"] = file.filename

    # Log for monitoring
    prediction_log.append({
        "timestamp": time.time(),
        "score": result["anomaly_score"],
        "is_defect": result["is_defect"],
    })

    logger.info(f"Prediction: {result['label']} | score={result['anomaly_score']} | {latency_ms}ms")
    return JSONResponse(content=result)


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict defect status for multiple images."""
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for file in files:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
        result = run_inference(tensor)
        result["filename"] = file.filename
        results.append(result)

    defect_count = sum(1 for r in results if r["is_defect"])
    return JSONResponse(content={
        "total": len(results),
        "defects_found": defect_count,
        "defect_rate": round(defect_count / len(results), 3),
        "predictions": results,
    })


@app.get("/stats")
async def stats():
    """Return defect rate statistics from prediction log."""
    if not prediction_log:
        return {"message": "No predictions yet"}

    scores = [p["score"] for p in prediction_log]
    defect_rate = sum(p["is_defect"] for p in prediction_log) / len(prediction_log)

    return {
        "total_predictions": len(prediction_log),
        "defect_rate": round(defect_rate, 4),
        "avg_anomaly_score": round(float(np.mean(scores)), 4),
        "max_anomaly_score": round(float(np.max(scores)), 4),
    }
