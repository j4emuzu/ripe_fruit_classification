"""
================================= Overview =================================

This service provides a REST API for Fruit Ripeness Classification.
It utilizes FastAPI for the web framework and YOLOv11 for the deep learning 
inference engine. The system processes images to identify 14 distinct 
classes representing 7 fruit types in 2 ripeness states.

============================================================================
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from ultralytics import YOLO
from PIL import Image
import io, os, httpx

app = FastAPI(title="Fruit Ripeness Classification API")

# Load embedded model
model = YOLO('best.pt')

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handles image uploads and returns ripeness status for 14 classes.
    Confidence threshold is set to 0.5 to ensure high-reliability output.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Image required.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model.predict(image, imgsz=224, conf=0.5)
    result = results[0]

    if result.probs is None:
        return {"message": "No fruit detected with sufficient confidence."}

    top1_idx = result.probs.top1
    class_name = result.names[top1_idx]
    confidence = result.probs.top1conf.item()

    status_raw, fruit_raw = class_name.split('_')

    return {
        "fruit_name": fruit_raw.capitalize(),
        "status": status_raw.capitalize(),
        "confidence_score": round(confidence, 4)
    }


@app.post("/chat")
async def chat(request: Request):
    """Proxy Gemini API — keeps API key server-side only."""
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    body = await request.json()
    async with httpx.AsyncClient() as client:
        res = await client.post(GEMINI_URL, json=body, timeout=30)
    return res.json()