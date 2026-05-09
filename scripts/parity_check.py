import json
import logging
from pathlib import Path
from PIL import Image

import numpy as np
import torch

from src.models.lit_module import DefectClassifier
from src.deployment.inference import ONNXInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = Path("C:/Users/manue/Downloads/archive (2)")
    models_onnx = root_dir / "models_onnx"
    
    with open(root_dir / "results" / "best_checkpoints.json", "r") as f:
        data = json.load(f)
        
    engine = ONNXInference(str(models_onnx))
    
    categories = ["bottle", "capsule", "carpet", "hazelnut", "leather", "pill"]
    
    all_passed = True
    
    for cat in categories:
        if cat not in engine.sessions:
            logger.warning(f"ONNX model for {cat} not found. Skipping parity check.")
            continue
            
        info = data.get(cat)
        if not info:
            continue
            
        selected_fold = info["selected_fold"]
        ckpt_path = root_dir / "checkpoints" / "supervised" / cat / f"fold_{selected_fold - 1}" / "best.ckpt"
        
        logger.info(f"Loading PyTorch model for {cat} from {ckpt_path}")
        pt_model = DefectClassifier.load_from_checkpoint(str(ckpt_path), map_location="cpu", pretrained=False)
        pt_model.eval()
        
        # Load 50 test images
        # We assume data is organized as archive (2)/<cat>/test/good or defect... 
        # For simplicity, we just rglob all images in archive (2)/<cat>
        cat_dir = data_dir / cat
        images_paths = list(cat_dir.rglob("*.png")) + list(cat_dir.rglob("*.jpg")) + list(cat_dir.rglob("*.jpeg"))
        
        # Take 50 images
        images_paths = images_paths[:50]
        if not images_paths:
            logger.warning(f"No images found for {cat} in {cat_dir}. Using dummy tensors for parity check.")
            images_paths = []
            
        if images_paths:
            logger.info(f"Validating {cat} on {len(images_paths)} images")
            pil_images = [Image.open(p) for p in images_paths]
            
            # Run ONNX inference
            # It expects RGB
            onnx_input = engine.preprocess(pil_images)
            
            pt_input = torch.tensor(onnx_input)
            
            with torch.no_grad():
                pt_logits = pt_model(pt_input).numpy()
                
            onnx_logits = engine.sessions[cat].run(None, {engine.sessions[cat].get_inputs()[0].name: onnx_input})[0]
            
            diff = np.abs(pt_logits - onnx_logits).max()
            logger.info(f"{cat} Max Diff: {diff:.2e}")
            if diff >= 1e-4:
                logger.error(f"❌ Parity check FAILED for {cat}: {diff:.2e} >= 1e-4")
                all_passed = False
            else:
                logger.info(f"✅ Parity check passed for {cat}")
        else:
            # Fallback to random tensors
            logger.info(f"Validating {cat} on 50 dummy random images")
            max_diff = 0.0
            session = engine.sessions[cat]
            input_name = session.get_inputs()[0].name
            for _ in range(50):
                x = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    pt_out = pt_model(x).cpu().numpy()
                onnx_out = session.run(None, {input_name: x.cpu().numpy()})[0]
                diff = float(np.abs(pt_out - onnx_out).max())
                max_diff = max(max_diff, diff)
            
            logger.info(f"{cat} Max Diff (dummy): {max_diff:.2e}")
            if max_diff >= 1e-4:
                logger.error(f"❌ Parity check FAILED for {cat}: {max_diff:.2e} >= 1e-4")
                all_passed = False
            else:
                logger.info(f"✅ Parity check passed for {cat} (dummy)")

    if all_passed:
        logger.info("ALL CATEGORIES PASSED ONNX PARITY VALIDATION.")
        
if __name__ == "__main__":
    main()
