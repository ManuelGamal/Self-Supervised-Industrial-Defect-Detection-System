import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

CATEGORY_THRESHOLDS = {
    "bottle": 0.3924,
    "capsule": 0.4337,
    "carpet": 0.4542,
    "hazelnut": 0.3982,
    "leather": 0.4731,
    "pill": 0.4564,
}

class ONNXInference:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.categories = ["bottle", "capsule", "carpet", "hazelnut", "leather", "pill"]
        self.thresholds = CATEGORY_THRESHOLDS
        self._load_models()

    def _load_models(self):
        for category in self.categories:
            model_path = self.model_dir / f"{category}.onnx"
            if model_path.exists():
                logger.info(f"Loading model for {category} from {model_path}")
                self.sessions[category] = ort.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
            else:
                logger.warning(f"Model not found for {category} at {model_path}")

    def preprocess(self, images: List[Image.Image]) -> np.ndarray:
        """
        Match PyTorch eval transform:
        Resize(224, 224)
        ToTensor() -> scales to [0, 1]
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        """
        processed = []
        for img in images:
            img = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_arr = (img_arr - mean) / std
            # HWC to CHW
            img_arr = np.transpose(img_arr, (2, 0, 1))
            processed.append(img_arr)
        
        return np.stack(processed, axis=0)

    def predict(self, category: str, images: List[Image.Image]) -> np.ndarray:
        if category not in self.sessions:
            raise ValueError(f"Model for category '{category}' not loaded.")
        
        batch = self.preprocess(images)
        session = self.sessions[category]
        input_name = session.get_inputs()[0].name
        
        logits = session.run(None, {input_name: batch})[0]
        # Apply sigmoid to get probability of class 1 (defect)
        # Using numerically stable sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
        # For class 1 probability: use logit[1] - logit[0] for proper 2-class
        probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
        return probs.reshape(-1, 1)

