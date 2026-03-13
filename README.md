# 🍎 Fruit Ripeness Classification

> AI inference service for classifying fruit ripeness — built with **FastAPI + YOLOv11**, fully containerized with Docker.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://www.docker.com/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-FF6B35)](https://github.com/ultralytics/ultralytics)

---

## 🚀 Overview

This service acts as the **"Brain"** within a distributed system — processing fruit images and returning precise ripeness classification results via a **Docker Bridge Network**.

The model classifies **14 classes** across 7 fruit types × 2 ripeness states (ripe / unripe):

| Fruit | Ripe Class | Unripe Class |
|-------|-----------|--------------|
| Apple | `ripe_apple` | `unripe_apple` |
| Banana | `ripe_banana` | `unripe_banana` |
| Mango | `ripe_mango` | `unripe_mango` |
| Orange | `ripe_orange` | `unripe_orange` |
| Papaya | `ripe_papaya` | `unripe_papaya` |
| Grape | `ripe_grape` | `unripe_grape` |
| Strawberry | `ripe_strawberry` | `unripe_strawberry` |

### Services

| Service | Container | Purpose |
|---------|-----------|---------|
| `web` | `fruit_web` | Nginx — serves frontend (`index.html`) |
| `api` | `fruit_api` | FastAPI + YOLOv11 — image inference |
| `npm` | `fruit_npm` | Nginx Proxy Manager — reverse proxy + SSL |

---

## 🏗️ Architecture

All services communicate over an internal Docker bridge network (`172.30.0.0/16`). Only the Nginx Proxy Manager exposes public ports.
```
Browser / Client
      │
      ▼  :80 (HTTP)  /  :443 (HTTPS)  /  :81 (Admin UI)
┌─────────────────────────┐
│  npm  (Nginx Proxy Mgr) │  ◄── fruit_npm
└────────────┬────────────┘
             │  app-network (172.30.0.0/16)
      ┌──────┴──────┐
      ▼             ▼
┌──────────┐  ┌───────────┐
│   web    │  │    api    │  ◄── fruit_api :8000
│  :80     │  │  :8000    │      FastAPI + YOLOv11
└──────────┘  └───────────┘
```

---

## 📦 Prerequisites

Ensure the following are installed on your system:

- [Git](https://git-scm.com/)
- [Docker Engine](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/) *(bundled with Docker Desktop)*

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/j4emuzu/ripe_fruit_classification.git
cd ripe_fruit_classification
```

> **Tip:** To access the AI service feature used for team development, use the dedicated branch:
> ```bash
> git clone -b ai/model-training https://github.com/j4emuzu/ripe_fruit_classification.git
> cd ripe_fruit_classification
> ```
> ⚠️ This branch is intended for internal testing and easier access to the AI service feature.

### 2. Configure Environment

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Launch the Service
```bash
docker compose up -d --build
```

> The first build may take a few minutes while Docker pulls base images and installs dependencies.
> Once complete, all containers run in the background.

### 4. Verify
```bash
docker compose ps
```

Expected output:
```
NAME          STATUS              PORTS
fruit_web     Up (healthy)        80/tcp
fruit_api     Up (healthy)        8000/tcp
fruit_npm     Up (healthy)        0.0.0.0:80->80, 443->443, 81->81
```

FastAPI interactive docs → **http://localhost:8000/docs**

---

## 🔌 API Reference

### `POST /predict`

Accepts an image file and returns the fruit name, ripeness status, and confidence score.

**Request**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/fruit.jpg"
```

**Response**
```json
{
  "fruit_name": "Mango",
  "status": "Ripe",
  "confidence_score": 0.9214
}
```

> Confidence threshold is set to `0.5` — results below this are discarded.

---

### `POST /chat`

Proxies requests to the Gemini API, keeping the API key server-side.

> Requires `GEMINI_API_KEY` to be set in your `.env` file.

---

### `GET /health`

Returns `200 healthy` when Nginx is running. Used by Docker health checks.

---

## ⚙️ Configuration

### Resource Limits

Defined in `docker-compose.yml`:

| Service | CPU Limit | Memory Limit | Memory Reservation |
|---------|-----------|--------------|--------------------|
| `web` | 0.25 cores | 64 MB | — |
| `api` | 2.0 cores | 4 GB | 2 GB |
| `npm` | 0.5 cores | 256 MB | — |

### Upload Size

Nginx accepts image uploads up to **32 MB** (`client_max_body_size 32M`).

---

## 🎮 GPU Acceleration (NVIDIA)

By default, the service runs on **CPU** for maximum compatibility. To enable NVIDIA GPU acceleration:

1. Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed on your host.

2. Open `docker-compose.yml` and locate:
```yaml
   # runtime: nvidia
```

3. Remove the `#` to enable:
```yaml
   runtime: nvidia
```

4. Restart the service:
```bash
   docker compose up -d
```

---

## 🗂️ Project Structure
```
ripe_fruit_classification/
├── Docker/
│   └── Dockerfile              # FastAPI + YOLOv11 image
├── ai/
│   ├── main.py                 # FastAPI app & YOLO inference
│   └── best.pt                 # Trained model weights
├── web/
│   └── index.html              # Frontend served by Nginx
├── npm/
│   ├── data/                   # Nginx Proxy Manager data
│   └── letsencrypt/            # SSL certificates
├── nginx.conf                  # Nginx configuration
├── docker-compose.yml          # Service orchestration
├── .env                        # Environment variables (not committed)
├── .dockerignore
└── .gitignore
```

---

## 🛠️ Development vs Production

The Dockerfile supports two modes, controlled by the `COPY` instructions:

**Development (default)**
- Code is served via volume mount (`./ai:/app`)
- Uvicorn runs with `--reload` — changes to `main.py` are picked up automatically, no rebuild needed

**Production**
- Uncomment the `COPY` lines in `Dockerfile` and remove the volume mount from `docker-compose.yml`
- Application code is baked into the image at build time
```dockerfile
# Uncomment for production:
COPY main.py .
COPY best.pt .
```

---

## 🧰 Common Commands
```bash
# Start all services
docker compose up -d --build

# Stop all services
docker compose down

# View live logs
docker compose logs -f

# Logs for a specific service
docker compose logs -f api

# Rebuild a single service
docker compose up -d --build api

# Check container status
docker compose ps

# Open a shell inside the api container
docker exec -it fruit_api bash
```

---

## 🐛 Troubleshooting

**Container exits immediately**
- Run `docker compose logs api` to inspect the error
- Confirm `best.pt` exists in the `./ai/` directory
- Ensure `.env` is present in the project root

**Port 80 / 443 already in use**
- Stop any running web server (Apache, Nginx, IIS) on the host
- Or change the port mapping in `docker-compose.yml`

**GPU not detected**
- Verify toolkit: `nvidia-container-cli --version`
- Confirm `runtime: nvidia` is uncommented in `docker-compose.yml`
- Restart the Docker daemon after installing the toolkit

**Low confidence / wrong predictions**
- Use clear, well-lit images with the fruit filling most of the frame
- Confidence threshold is `0.5` in `main.py` — adjust if needed
- Supported: Apple, Banana, Mango, Orange, Papaya, Grape, Strawberry

---

## 👥 Team

**MercedesBenz Team** — Fruit Ripeness Classification v1.0
Thanpisit Banyam
Thoranin Akkaratham
Chanatip Ruanjaiman
---

*FastAPI + YOLOv11 · Docker · Gemini AI*