# ╔════════════════════════════════════════════════════════════════════╗
# ║         Ripe Fruit Classification - FastAPI + YOLO Dockerfile      ║
# ║                                                                    ║
# ║  Base Image: Python 3.10 (Official, slim variant)                  ║
# ║  Dependencies: FastAPI, Uvicorn, Ultralytics (YOLOv11+)            ║
# ║  Model: best.pt (placed in ./ai/ before build)                     ║
# ╚════════════════════════════════════════════════════════════════════╝

FROM python:3.10-slim

LABEL maintainer="MercedesBenz Team"
LABEL description="FastAPI + YOLOv11 Fruit Ripeness Classification"
LABEL version="1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ─────────────────────────────────────────────────────────────────────
# SYSTEM DEPENDENCIES
# Note: libatlas-base-dev removed from Debian Trixie — libopenblas-dev ทำหน้าที่แทน
# ─────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libharfbuzz0b \
    libwebp7 \
    libtiff-dev \
    libjpeg-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─────────────────────────────────────────────────────────────────────
# INSTALL PYTHON PACKAGES
# ultralytics>=8.3.0 รองรับ YOLOv11 และ C3k2 module
# ใช้ >= แทน == เพื่อให้ pip จัดการ dependency conflicts เอง
# ─────────────────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    "ultralytics>=8.3.0" \
    "pillow>=10.1.0" \
    "numpy>=1.24.3" \
    "opencv-python-headless>=4.8.1.78" \
    "torch>=2.1.1" \
    "torchvision>=0.16.1" \
    requests==2.31.0 \
    python-multipart

RUN pip install httpx

# ─────────────────────────────────────────────────────────────────────
# COPY APPLICATION CODE
# DEV  : comment ไว้ — ใช้ volume mount ใน docker-compose.yml แทน
#        แก้ main.py แล้ว uvicorn --reload จะ reload อัตโนมัติ
# PROD : เปิด COPY ทั้งสองบรรทัด แล้วลบ volume ./ai:/app ออกจาก compose
# ─────────────────────────────────────────────────────────────────────
# COPY main.py .
# COPY best.pt .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]