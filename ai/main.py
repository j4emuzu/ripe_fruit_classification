"""
================================= Overview =================================

This service provides a REST API for Fruit Ripeness Classification.
It utilizes FastAPI for the web framework and YOLOv26 for the deep learning 
inference engine. The system processes images to identify 14 distinct 
classes representing 7 fruit types in 2 ripeness states.

============================================================================
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI(title="Fruit Ripeness Classification API")

# Load embedded model
# Model is stored locally within the container for immediate availability
model = YOLO('best.pt') 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handles image uploads and returns ripeness status for 14 classes.
    Confidence threshold is set to 0.5 to ensure high-reliability output.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Image required.")

    # Process uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Inference logic
    results = model.predict(image, imgsz=224, conf=0.5)
    result = results[0]

    if result.probs is None:
        return {"message": "No fruit detected with sufficient confidence."}

    # Extract top prediction and confidence
    top1_idx = result.probs.top1
    class_name = result.names[top1_idx]
    confidence = result.probs.top1conf.item()

    # Parse status and fruit type from 14-class label
    status_raw, fruit_raw = class_name.split('_')

    return {
        "fruit_name": fruit_raw.capitalize(),
        "status": status_raw.capitalize(),
        "confidence_score": round(confidence, 4),   # Decimal for logic
        "confidence_display": f"{confidence:.2%}",  # Percentage for UI
        "raw_label": class_name
    }