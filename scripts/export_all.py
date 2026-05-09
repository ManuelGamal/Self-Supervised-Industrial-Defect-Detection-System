import json
import os
import subprocess
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    checkpoints_dir = root_dir / "checkpoints" / "supervised"
    results_json = root_dir / "results" / "best_checkpoints.json"
    output_dir = root_dir / "models_onnx"
    output_dir.mkdir(exist_ok=True)

    with open(results_json, "r") as f:
        data = json.load(f)

    categories = ["bottle", "capsule", "carpet", "hazelnut", "leather", "pill"]
    
    for cat in categories:
        info = data.get(cat)
        if not info:
            print(f"Skipping {cat}, no info found.")
            continue
        
        # Directory structure is checkpoints/supervised/<cat>/fold_<selected_fold - 1>/best.ckpt
        selected_fold = info["selected_fold"]
        ckpt_path = checkpoints_dir / cat / f"fold_{selected_fold - 1}" / "best.ckpt"
        
        if not ckpt_path.exists():
            print(f"Error: Checkpoint not found at {ckpt_path}")
            continue
            
        out_path = output_dir / f"{cat}.onnx"

        
        print(f"Exporting {cat} from {ckpt_path} to {out_path}...")
        
        # modify src/models/export_onnx.py to test actual parity?
        # The instructions say: "max abs diff <1e-4 vs. PyTorch on 50 test images per category"
        # Since I'll run parity later, I'll just use the script to export.
        
        import sys
        cmd = [
            sys.executable,
            "-m", "src.models.export_onnx",
            "--checkpoint", str(ckpt_path),
            "--output", str(out_path)
        ]
        
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
