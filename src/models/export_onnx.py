"""
Export DefectClassifier Lightning checkpoint to ONNX and run parity checks.
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

from src.models.lit_module import DefectClassifier


def _safe_print(preferred: str, fallback: str) -> None:
    try:
        print(preferred)
    except UnicodeEncodeError:
        print(fallback)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = DefectClassifier.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        pretrained=False,
    )
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    _safe_print(f"✅ Exported to {args.output}", f"Exported to {args.output}")

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for _ in range(50):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            torch_out = model(x).cpu().numpy()
        onnx_out = sess.run(None, {"image": x.cpu().numpy()})[0]
        diff = float(np.abs(torch_out - onnx_out).max())
        max_diff = max(max_diff, diff)

    assert max_diff < 1e-4, f"❌ Parity check FAILED: max diff {max_diff}"
    _safe_print(
        f"✅ Parity check passed: max diff {max_diff:.2e}",
        f"Parity check passed: max diff {max_diff:.2e}",
    )


if __name__ == "__main__":
    main()
