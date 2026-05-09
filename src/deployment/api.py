import io
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from src.deployment.inference import ONNXInference, CATEGORY_THRESHOLDS

app = FastAPI(title="Defect Detection API", description="API for ONNX Models")

engine: ONNXInference = None

@app.on_event("startup")
def startup_event():
    global engine
    engine = ONNXInference("models_onnx")

@app.get("/health")
def health_check():
    if not engine or not engine.sessions:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "models not loaded"})
    return {"status": "healthy"}

@app.get("/thresholds")
def get_thresholds():
    return {"thresholds": CATEGORY_THRESHOLDS}

@app.get("/metrics")
def get_metrics():
    return {"latency": "measured via load testing"}

@app.post("/predict")
async def predict(
    category: str = Form(...),
    file: UploadFile = File(...)
):
    global engine
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        start_time = time.time()
        probs = engine.predict(category, [image])
        latency = time.time() - start_time
        
        threshold = CATEGORY_THRESHOLDS.get(category, 0.5)
        score = float(probs[0][0])
        is_defect = score >= threshold
        
        return {
            "category": category,
            "anomaly_score": round(score, 4),
            "threshold": threshold,
            "is_defect": is_defect,
            "label": "defect" if is_defect else "normal",
            "latency_ms": round(latency * 1000, 2)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(
    category: str = Form(...),
    files: List[UploadFile] = File(...)
):
    global engine
    try:
        images = []
        for f in files:
            contents = await f.read()
            images.append(Image.open(io.BytesIO(contents)).convert("RGB"))
        
        start_time = time.time()
        probs = engine.predict(category, images)
        latency = time.time() - start_time
        
        threshold = CATEGORY_THRESHOLDS.get(category, 0.5)
        
        results = []
        for f, p in zip(files, probs):
            score = float(p[0])
            is_defect = score >= threshold
            results.append({
                "filename": f.filename,
                "anomaly_score": round(score, 4),
                "is_defect": is_defect,
                "label": "defect" if is_defect else "normal"
            })
        
        defect_count = sum(1 for r in results if r["is_defect"])
        
        return {
            "category": category,
            "threshold": threshold,
            "total": len(results),
            "defects_found": defect_count,
            "results": results,
            "latency_ms": round(latency * 1000, 2)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
