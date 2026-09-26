# Deployment Guide

This guide covers deploying IndicOCR in various environments.

## Deployment Options

| Method | Use Case | Complexity |
|--------|----------|------------|
| **Local/Colab** | Development, testing | Low |
| **Docker** | Consistent environments, CI/CD | Medium |
| **REST API** | Integration with apps | Medium |
| **Cloud (AWS/GCP/Azure)** | Production, scaling | High |
| **Edge/Offline** | No internet, privacy | High |

---

## 1. Local Development

### Requirements
- Python 3.9+
- CUDA 11.8+ (for GPU)
- 8GB+ RAM, 4GB+ VRAM (recommended)

### Setup
```bash
git clone https://github.com/yourteam/indic-ocr-system.git
cd indic-ocr-system

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -e ".[dev]"

# Download models
python scripts/download_models.py --all

# Test
python examples/demo.py --image data/sample.jpg
```

### GPU Setup (Linux)
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install PaddlePaddle GPU
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu121/
```

---

## 2. Docker Deployment

### Dockerfile (included)
```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tesseract languages
RUN apt-get update && apt-get install -y \
    tesseract-ocr-hin tesseract-ocr-ben tesseract-ocr-tel \
    tesseract-ocr-mar tesseract-ocr-tam tesseract-ocr-guj \
    tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-pan \
    tesseract-ocr-ori tesseract-ocr-asm tesseract-ocr-urd \
    tesseract-ocr-eng tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Download models
RUN python scripts/download_models.py --all

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "examples/api_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run
```bash
# Build
docker build -t indic-ocr:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data indic-ocr:latest

# Run CPU only
docker run -p 8000:8000 -v $(pwd)/data:/app/data indic-ocr:latest

# Run with custom config
docker run -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  indic-ocr:latest
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  indic-ocr:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./configs:/app/configs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped

  # Optional: Redis for caching
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  # Optional: Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - indic-ocr
```

```bash
docker-compose up -d
```

---

## 3. REST API Deployment

### Using the Built-in FastAPI Server
```bash
# Development
python examples/api_server.py --reload

# Production
gunicorn examples.api_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### API Usage
```bash
# Single image
curl -X POST "http://localhost:8000/ocr" \
  -H "accept: application/json" \
  -F "file=@land_record.jpg" \
  -F "languages=hi,en,bn" \
  -F "return_visualization=true"

# Base64
curl -X POST "http://localhost:8000/ocr/base64" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "image_base64=$(base64 -w0 image.jpg)&return_structured=true"

# Batch
curl -X POST "http://localhost:8000/ocr/batch" \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "max_workers=4"
```

### Python Client
```python
import requests

# Single
with open("image.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/ocr",
        files={"file": f},
        data={"languages": "hi,en", "return_structured": "true"}
    )
print(resp.json()["text"])

# Batch
files = [("files", open(f, "rb")) for f in ["1.jpg", "2.jpg"]]
resp = requests.post("http://localhost:8000/ocr/batch", files=files)
for r in resp.json()["results"]:
    print(r["text"])
```

### API Scaling
```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: indic-ocr-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: indic-ocr-api
  template:
    metadata:
      labels:
        app: indic-ocr-api
    spec:
      containers:
      - name: api
        image: indic-ocr:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "4Gi"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
---
apiVersion: v1
kind: Service
metadata:
  name: indic-ocr-service
spec:
  selector:
    app: indic-ocr-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 4. Cloud Deployment

### AWS

#### Option A: ECS/Fargate (CPU only)
```json
{
  "family": "indic-ocr",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "containerDefinitions": [{
    "name": "indic-ocr",
    "image": "your-account.dkr.ecr.region.amazonaws.com/indic-ocr:latest",
    "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/indic-ocr",
        "awslogs-region": "us-east-1"
      }
    }
  }]
}
```

#### Option B: EC2 with GPU (p3/g4/g5 instances)
```bash
# Launch p3.2xlarge (1 V100) or g5.xlarge (1 A10G)
# Use Deep Learning AMI
# Install Docker, NVIDIA Container Toolkit
# Run docker-compose
```

#### Option C: SageMaker Endpoint
```python
# Deploy as SageMaker real-time endpoint
import sagemaker
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    framework_version="2.1",
    py_version="py39"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="indic-ocr-endpoint"
)
```

### GCP

#### Cloud Run (CPU)
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/indic-ocr

# Deploy
gcloud run deploy indic-ocr \
  --image gcr.io/PROJECT_ID/indic-ocr \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --allow-unauthenticated
```

#### Vertex AI (GPU)
```bash
# Custom container on Vertex AI
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=indic-ocr-training \
  --worker-pool-spec=machine-type=a2-highgpu-1g,replica-count=1,container-image-uri=gcr.io/PROJECT_ID/indic-ocr-training
```

### Azure

#### Container Instances
```bash
az container create \
  --resource-group myRG \
  --name indic-ocr \
  --image yourregistry.azurecr.io/indic-ocr:latest \
  --gpu-count 1 \
  --gpu-type NvidiaTeslaT4 \
  --memory 8 \
  --cpu 4 \
  --ports 8000 \
  --ip-address Public
```

---

## 5. Edge/Offline Deployment

### Requirements
- All models pre-downloaded
- No internet access needed
- Tesseract traineddata local

### Preparation
```bash
# 1. Download all models on internet-connected machine
python scripts/download_models.py --all

# 2. Verify
python scripts/download_models.py --verify

# 3. Package for transfer
tar -czf indic-ocr-offline.tar.gz models/ configs/ src/
```

### Target Machine Setup
```bash
# 1. Extract
tar -xzf indic-ocr-offline.tar.gz

# 2. Install system dependencies (offline)
# Download .deb/.rpm packages for:
# - tesseract-ocr + all language packs
# - libopencv, libglib2.0, libgl1
# Install: dpkg -i *.deb

# 3. Install Python packages (offline)
# On online machine: pip download -r requirements.txt -d ./packages
# Transfer packages/ folder
# On target: pip install --no-index --find-links ./packages -r requirements.txt

# 4. Set environment
export INDIC_OCR_MODELS_DIR=/path/to/models
export TESSDATA_PREFIX=/path/to/models/tesseract/tessdata

# 5. Run
python examples/demo.py --image test.jpg
```

### Raspberry Pi / Jetson Nano
```bash
# Use lighter models
# - PaddleOCR mobile models
# - TrOCR tiny (if available) or skip
# - Tesseract only

# Build for ARM64
docker build --platform linux/arm64 -t indic-ocr-arm .
```

---

## 6. Performance Optimization

### Model Optimization

#### ONNX Export
```python
# Export PaddleOCR to ONNX
import paddle
paddle.onnx.export(model, "model.onnx")

# Export TrOCR to ONNX
from optimum.onnxruntime import ORTModelForVision2Seq
model = ORTModelForVision2Seq.from_pretrained("models/trocr-finetuned", export=True)
model.save_pretrained("models/trocr-onnx")
```

#### TensorRT (NVIDIA)
```python
# Convert ONNX to TensorRT
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

#### Quantization
```python
# Dynamic quantization (CPU)
import torch
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Inference Optimization

#### Batch Processing
```python
# Process multiple images together
results = ocr.process_batch(image_list, max_workers=4)
```

#### Model Warmup
```python
# Warmup on startup
dummy = np.ones((640, 640, 3), dtype=np.uint8) * 255
for _ in range(3):
    _ = ocr.process(dummy)
```

#### Caching
```python
# Cache frequent predictions
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_ocr(image_hash):
    return ocr.process(image_hash)
```

### Monitoring

#### Metrics to Track
- Latency (p50, p95, p99)
- Throughput (req/s)
- GPU utilization
- Memory usage
- Error rate
- Accuracy (periodic)

#### Prometheus/Grafana
```yaml
# Add to FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

---

## 7. Security Considerations

### Input Validation
```python
# Limit file size
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Validate image
def validate_image(file: UploadFile):
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
    # Check magic bytes
    header = await file.read(8)
    await file.seek(0)
    if not is_valid_image(header):
        raise HTTPException(400, "Invalid image")
```

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/ocr")
@limiter.limit("30/minute")
async def ocr_endpoint(request: Request, ...):
    ...
```

### Authentication
```python
# API Key
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key not in VALID_KEYS:
        raise HTTPException(401, "Invalid API key")
```

---

## 8. CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/ci.yml
name: CI/CD

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.10'}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
      - run: ruff check src/
      - run: black --check src/
      - run: mypy src/

  docker:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ secrets.DOCKER_USERNAME }}/indic-ocr:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: docker
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # kubectl set image deployment/indic-ocr api=...
```

---

## 9. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch size, enable gradient accumulation, use fp16 |
| Tesseract not found | Install tesseract-ocr system package, set TESSDATA_PREFIX |
| PaddleOCR slow | Enable GPU, use MKL-DNN, reduce image size |
| fastText model missing | Run `download_models.py --fasttext` |
| Import errors | Check Python version, reinstall with `pip install -e .` |
| Memory leak | Clear GPU cache: `torch.cuda.empty_cache()` |

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python examples/demo.py --image test.jpg
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Model status
curl http://localhost:8000/ | jq '.models_loaded'
```

---

## 10. Backup & Recovery

### Model Backup
```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Store in S3/GCS
aws s3 cp models_backup.tar.gz s3://bucket/backups/
```

### Config Backup
```bash
# Version control configs
git add configs/
git commit -m "config: update model parameters"
```

---

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: team@sih-hackathon.com
- **Wiki**: Project Wiki for detailed guides