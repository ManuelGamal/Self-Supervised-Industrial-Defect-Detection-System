#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint.ckpt> <output.onnx>"}
OUTPUT=${2:?"Usage: $0 <checkpoint.ckpt> <output.onnx>"}

python -m src.models.export_onnx --checkpoint "$CHECKPOINT" --output "$OUTPUT"
