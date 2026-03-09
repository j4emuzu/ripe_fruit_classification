# Base Image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies for YOLO and FastAPI
RUN pip install --no-cache-dir \
    ultralytics \
    fastapi \
    uvicorn \
    python-multipart \
    opencv-python-headless

# Embed the latest trained v5 model and API code
COPY ai/best.pt /app/best.pt
COPY ai/main.py /app/main.py

# Expose Port 8000 for internal Docker Bridge network communication
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]