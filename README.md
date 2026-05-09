#  Self-Supervised Industrial Defect Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](docker/)

A production-ready industrial defect detection system that leverages **self-supervised learning** to learn rich visual representations from unlabeled industrial images, enabling high-quality defect detection with minimal labeled data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Milestones](#milestones)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Results](#results)
- [Contributing](#contributing)

---

## Overview

Industrial defect detection is challenging due to the scarcity of labeled defect images. This project addresses that by:

1. **Self-Supervised Pretraining** — Learning visual representations from unlabeled normal images using contrastive/self-distillation methods (SimCLR, BYOL, DINO)
2. **Efficient Fine-tuning** — Leveraging pretrained encoders for defect detection with limited labels
3. **Production Deployment** — Containerized REST API with ONNX optimization, monitoring, and drift detection

### Key Features

-  SSL pretraining with SimCLR / BYOL / DINO
-  Anomaly detection via feature distance scoring
-  Pixel-wise segmentation with U-Net decoder
-  ONNX export + quantization for low-latency inference
-  Docker containerized REST API
-  Real-time monitoring & drift detection dashboard
-  Automated retraining pipeline

---

## Project Structure

```
defect-detection/
├── configs/                    # Hydra/YAML config files
│   ├── ssl_pretrain.yaml
│   ├── finetune.yaml
│   └── deploy.yaml
├── data/
│   ├── raw/                    # Raw downloaded datasets
│   ├── processed/              # Preprocessed images
│   └── splits/                 # Train/val/test split CSVs
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/                       # Documentation and reports
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_SSL_Pretraining.ipynb
│   ├── 03_Finetuning.ipynb
│   ├── 04_Evaluation.ipynb
│   └── 05_Deployment_Demo.ipynb
├── scripts/
│   ├── download_data.sh
│   ├── train_ssl.sh
│   └── export_onnx.sh
├── src/
│   ├── data/                   # Dataset classes & preprocessing
│   ├── models/                 # Encoder, detection, segmentation heads
│   ├── ssl/                    # SSL frameworks (SimCLR, BYOL, DINO)
│   ├── evaluation/             # Metrics and evaluation logic
│   ├── deployment/             # FastAPI app & inference engine
│   └── monitoring/             # Drift detection & logging
└── tests/                      # Unit and integration tests
```

---

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Data Collection, Preprocessing & EDA | ⬜ Todo |
| 2 | Self-Supervised Representation Learning | ⬜ Todo |
| 3 | Defect Detection & Segmentation | ⬜ Todo |
| 4 | Deployment, Optimization & MLOps | ⬜ Todo |
| 5 | Final Documentation & Presentation | ⬜ Todo |

---

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- Docker & Docker Compose

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/defect-detection.git
cd defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Quick Start

```bash
# 1. Download datasets
bash scripts/download_data.sh

# 2. Run SSL pretraining
bash scripts/train_ssl.sh --config configs/ssl_pretrain.yaml

# 3. Fine-tune for defect detection
python src/models/train_finetune.py --config configs/finetune.yaml

# 4. Launch inference API
docker-compose up --build
```

---

## Datasets

| Dataset | Type | Classes | Images |
|---------|------|---------|--------|
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Anomaly Detection | 15 | ~5000 |
| [DAGM](https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection) | Texture Defects | 10 | ~6000 |
| [NEU Surface Defect](http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm) | Steel Surface | 6 | 1800 |

Download instructions: see [`scripts/download_data.sh`](scripts/download_data.sh)

---

## Training

### SSL Pretraining

```bash
python src/ssl/train_ssl.py \
  --method simclr \
  --backbone resnet50 \
  --epochs 200 \
  --batch_size 256 \
  --temperature 0.07 \
  --data_dir data/processed/normal/
```

Supported methods: `simclr`, `byol`, `dino`

### Fine-tuning

```bash
python src/models/train_finetune.py \
  --encoder_checkpoint checkpoints/ssl_encoder.pth \
  --mode anomaly  # or: classification, segmentation
  --labeled_fraction 0.1
```

---

## Evaluation

```bash
python src/evaluation/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --test_dir data/splits/test/

# Outputs: AUROC, F1, mAP, Pixel IoU
```

---

## Deployment

### Run Inference API

```bash
docker-compose up --build
```

API available at `http://localhost:8000`

```bash
# Single image inference
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_image.jpg"

# Batch inference
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@img1.jpg" -F "files=@img2.jpg"
```

### Export to ONNX

```bash
bash scripts/export_onnx.sh --checkpoint checkpoints/best_model.pth
```

---

## Results

> Results will be populated after training runs.

| Model | AUROC | F1 | Inference Latency |
|-------|-------|----|-------------------|
| SSL Pretrained (SimCLR) | - | - | - |
| SSL Pretrained (BYOL) | - | - | - |
| Fully Supervised Baseline | - | - | - |
| Random Init Baseline | - | - | - |

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push and open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE)
