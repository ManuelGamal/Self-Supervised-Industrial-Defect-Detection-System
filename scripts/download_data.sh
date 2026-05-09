#!/bin/bash
# Download and prepare industrial defect detection datasets.
# Usage: bash scripts/download_data.sh [--dataset mvtec|neu|all]

set -e

DATASET=${1:-all}
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "=== Industrial Defect Detection Dataset Downloader ==="

download_mvtec() {
    echo ""
    echo ">>> Downloading MVTec AD Dataset..."
    echo "NOTE: MVTec AD requires manual download due to license agreement."
    echo "1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad"
    echo "2. Accept the license and download mvtec_anomaly_detection.tar.xz"
    echo "3. Extract to: data/raw/mvtec/"
    echo ""
    echo "Expected structure:"
    echo "  data/raw/mvtec/"
    echo "    bottle/train/good/*.png"
    echo "    bottle/test/broken_large/*.png"
    echo "    ..."
}

download_neu() {
    echo ""
    echo ">>> Downloading NEU Surface Defect Dataset..."
    echo "NOTE: NEU dataset requires registration."
    echo "1. Visit: http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm"
    echo "2. Download and extract to: data/raw/neu/"
    echo ""
    echo "Expected structure:"
    echo "  data/raw/neu/"
    echo "    train/{Crazing,Inclusion,Patches,Pitted,Rolled,Scratches}/*.jpg"
    echo "    test/{Crazing,Inclusion,Patches,Pitted,Rolled,Scratches}/*.jpg"
}

case "$DATASET" in
    mvtec)  download_mvtec ;;
    neu)    download_neu ;;
    all)    download_mvtec; download_neu ;;
    *)      echo "Unknown dataset: $DATASET. Options: mvtec, neu, all" && exit 1 ;;
esac

echo ""
echo "=== After downloading datasets, run preprocessing: ==="
echo "  python src/data/preprocess.py --dataset mvtec --input data/raw/mvtec --output data/processed"
