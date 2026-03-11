# ╔════════════════════════════════════════════════════════════════════╗
# ║         Ripe Fruit Classification - FastAPI + YOLO Dockerfile      ║
# ║                                                                    ║
# ║  Base Image: Python 3.10 (Official, slim variant)                  ║
# ║  Dependencies: FastAPI, Uvicorn, Ultralytics (YOLOv8)              ║
# ║  Model: best.pt (placed in ./ai/ before build)                     ║
# ╚════════════════════════════════════════════════════════════════════╝

FROM python:3.10-slim

# ─────────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────────
LABEL maintainer="MercedesBenz Team"
LABEL description="FastAPI + YOLOv8 Fruit Ripeness Classification"
LABEL version="1.0"

# ─────────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKDIR=/app

# ─────────────────────────────────────────────────────────────────────
# SYSTEM DEPENDENCIES (for image processing + YOLO)
# ─────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libharfbuzz0b \
    libwebp6 \
    libtiff6 \
    libjasper1 \
    libjpeg62-turbo \
    libatlas-base-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────
# SET WORKING DIRECTORY
# ─────────────────────────────────────────────────────────────────────
WORKDIR ${WORKDIR}

# ─────────────────────────────────────────────────────────────────────
# COPY REQUIREMENTS & INSTALL PYTHON PACKAGES
# ─────────────────────────────────────────────────────────────────────
# Create requirements.txt (or copy if exists)
RUN echo "fastapi==0.104.1\n\
    uvicorn[standard]==0.24.0\n\
    ultralytics==8.0.206\n\
    pillow==10.1.0\n\
    numpy==1.24.3\n\
    opencv-python-headless==4.8.1.78\n\
    torch==2.1.1\n\
    torchvision==0.16.1\n\
    requests==2.31.0" > requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────
# COPY APPLICATION CODE
# ─────────────────────────────────────────────────────────────────────
# Copy main.py and best.pt model
COPY main.py .
COPY best.pt .

# ─────────────────────────────────────────────────────────────────────
# EXPOSE PORT
# ─────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# ─────────────────────────────────────────────────────────────────────
# RUN APPLICATION
# ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
