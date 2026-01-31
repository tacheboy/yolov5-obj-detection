"""
YOLOv5 FastAPI Inference Service

A REST API for running object detection on uploaded images.
Run from project root: uvicorn api.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /           - Health check and API info
    GET  /model      - Model information
    POST /predict    - Run inference on uploaded image
"""

import io
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.model import load_model, get_model, YOLOv5ONNX


# Pydantic models for API responses
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str


class PredictionResponse(BaseModel):
    success: bool
    num_detections: int
    detections: List[Detection]
    image_size: dict


class ModelInfo(BaseModel):
    model_path: str
    input_shape: list
    img_size: int
    provider: str
    conf_threshold: float
    iou_threshold: float
    num_classes: int
    class_names: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# Create FastAPI app
app = FastAPI(
    title="YOLOv5 Object Detection API",
    description="REST API for YOLOv5 object detection inference using ONNX Runtime",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model instance
model: Optional[YOLOv5ONNX] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    try:
        # Get configuration from environment variables
        model_path = os.environ.get("MODEL_PATH", None)
        conf_threshold = float(os.environ.get("CONF_THRESHOLD", "0.25"))
        iou_threshold = float(os.environ.get("IOU_THRESHOLD", "0.45"))
        use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
        
        # Class names (comma-separated in env var, or default)
        class_names_env = os.environ.get("CLASS_NAMES", "")
        class_names = class_names_env.split(",") if class_names_env else None
        
        model = load_model(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            use_gpu=use_gpu,
        )
        print(f"Model loaded successfully: {model.model_path}")
        print(f"Using provider: {model.provider}")
    except Exception as e:
        print(f"Warning: Failed to load model on startup: {e}")
        print("Model will need to be loaded manually or API will return errors")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.get("/model", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the ONNX model exists at artifacts/model.onnx",
        )
    
    info = model.get_info()
    return ModelInfo(**info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file to run detection on"),
    conf_threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold (overrides model default)",
    ),
    iou_threshold: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS (overrides model default)",
    ),
):
    """
    Run object detection on an uploaded image.
    
    - **file**: Image file (JPEG, PNG, etc.)
    - **conf_threshold**: Optional confidence threshold override
    - **iou_threshold**: Optional IoU threshold override for NMS
    
    Returns detected objects with bounding boxes, confidence scores, and class names.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the ONNX model exists at artifacts/model.onnx",
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file.",
        )
    
    try:
        # Read image data
        contents = await file.read()
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image. Please ensure the file is a valid image.",
            )
        
        # Override thresholds if provided
        original_conf = model.conf_threshold
        original_iou = model.iou_threshold
        
        try:
            if conf_threshold is not None:
                model.conf_threshold = conf_threshold
            if iou_threshold is not None:
                model.iou_threshold = iou_threshold
            
            # Run inference
            detections = model.predict(image)
        finally:
            # Restore original thresholds
            model.conf_threshold = original_conf
            model.iou_threshold = original_iou
        
        return PredictionResponse(
            success=True,
            num_detections=len(detections),
            detections=[Detection(**det) for det in detections],
            image_size={"width": image.shape[1], "height": image.shape[0]},
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
