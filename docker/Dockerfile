FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    onnxruntime \
    Pillow \
    numpy \
    torchvision \
    python-multipart \
    prometheus-client

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/

# Create checkpoint and log directories
RUN mkdir -p checkpoints logs

EXPOSE 8000

CMD ["uvicorn", "src.deployment.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
