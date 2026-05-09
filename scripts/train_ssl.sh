#!/bin/bash
# SSL Pretraining Script
# Usage: bash scripts/train_ssl.sh [--method simclr] [--backbone resnet50] [--epochs 200]

set -e

METHOD=${METHOD:-simclr}
BACKBONE=${BACKBONE:-resnet50}
EPOCHS=${EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-256}
CONFIG=${CONFIG:-configs/ssl_pretrain.yaml}

echo "=== SSL Pretraining ==="
echo "Method:    $METHOD"
echo "Backbone:  $BACKBONE"
echo "Epochs:    $EPOCHS"
echo "Config:    $CONFIG"
echo ""

python src/ssl/train_ssl.py \
    --config "$CONFIG"

echo ""
echo "=== Pretraining complete ==="
echo "Encoder saved to: checkpoints/ssl/best_encoder.pth"
echo ""
echo "Next step - fine-tune for defect detection:"
echo "  python src/models/train_finetune.py --config configs/finetune.yaml"
