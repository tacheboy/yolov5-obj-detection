"""
YOLOv5 ONNX Model Loader and Inference

Provides model loading and inference utilities for the FastAPI service.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

def select_onnx_providers(use_gpu: bool) -> List[str]:
    """Select ONNX Runtime execution providers based on availability."""
    if not ONNX_AVAILABLE:
        return []
    available = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class YOLOv5ONNX:
    """YOLOv5 ONNX model wrapper for inference."""
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[List[str]] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize the YOLOv5 ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            class_names: List of class names (optional)
            use_gpu: Whether to use GPU if available
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime is not installed. Run: pip install onnxruntime")
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []
        
        # Select execution providers
        providers = select_onnx_providers(use_gpu)
        if use_gpu and "CUDAExecutionProvider" not in providers:
            print("Warning: CUDA provider not available. Falling back to CPU.")
        
        # Load model
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Determine input size (typically [1, 3, 640, 640])
        if len(self.input_shape) == 4:
            self.img_size = self.input_shape[2]  # Height
        else:
            self.img_size = 640  # Default
        
        # Get active provider
        self.provider = self.session.get_providers()[0]
    
    def preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image in BGR format (OpenCV)
        
        Returns:
            Tuple of (preprocessed_image, scale_factor, original_size, padding)
        """
        original_h, original_w = image.shape[:2]
        
        # Calculate scale to preserve aspect ratio
        scale = min(self.img_size / original_w, self.img_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (letterbox)
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize and transpose
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)  # HWC to CHW
        batched = np.expand_dims(transposed, axis=0)  # Add batch dimension
        
        return batched, scale, (original_w, original_h), (pad_x, pad_y)
    
    def postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        original_size: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to get detections.
        
        Args:
            outputs: Raw model outputs
            scale: Scale factor used in preprocessing
            original_size: Original image size (width, height)
            padding: Padding applied (pad_x, pad_y)
        
        Returns:
            List of detection dictionaries
        """
        # YOLOv5 output shape: [1, num_detections, 5 + num_classes]
        # Each detection: [x_center, y_center, width, height, obj_conf, class1_conf, class2_conf, ...]
        
        predictions = outputs[0] if isinstance(outputs, list) else outputs
        
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
        
        detections = []
        pad_x, pad_y = padding
        original_w, original_h = original_size
        
        for pred in predictions:
            obj_conf = pred[4]
            
            if obj_conf < self.conf_threshold:
                continue
            
            # Get class scores
            class_scores = pred[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = class_scores[class_id]
            
            # Combined confidence
            confidence = obj_conf * class_conf
            
            if confidence < self.conf_threshold:
                continue
            
            # Get bounding box (center format to corner format)
            cx, cy, w, h = pred[:4]
            
            # Remove padding offset
            cx = cx - pad_x
            cy = cy - pad_y
            
            # Scale back to original image size
            cx = cx / scale
            cy = cy / scale
            w = w / scale
            h = h / scale
            
            # Convert to corner format
            x1 = max(0, cx - w / 2)
            y1 = max(0, cy - h / 2)
            x2 = min(original_w, cx + w / 2)
            y2 = min(original_h, cy + h / 2)
            
            detection = {
                "box": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
                "confidence": float(confidence),
                "class_id": class_id,
                "class_name": self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
            }
            detections.append(detection)
        
        # Apply NMS
        if detections:
            detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                det for det in detections
                if self._iou(best["box"], det["box"]) < self.iou_threshold
                or det["class_id"] != best["class_id"]
            ]
        
        return keep
    
    def _iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1["x1"], box2["x1"])
        y1 = max(box1["y1"], box2["y1"])
        x2 = min(box1["x2"], box2["x2"])
        y2 = min(box1["y2"], box2["y2"])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
        area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on an image.
        
        Args:
            image: Input image in BGR format (OpenCV)
        
        Returns:
            List of detection dictionaries
        """
        # Preprocess
        input_tensor, scale, original_size, padding = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        detections = self.postprocess(outputs[0], scale, original_size, padding)
        
        return detections
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_path": str(self.model_path),
            "input_shape": self.input_shape,
            "img_size": self.img_size,
            "provider": self.provider,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
        }


# Global model instance (loaded once at startup)
_model: Optional[YOLOv5ONNX] = None


def load_model(
    model_path: str = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_names: List[str] = None,
    use_gpu: bool = True,
) -> YOLOv5ONNX:
    """
    Load the YOLOv5 ONNX model (singleton pattern).
    
    Args:
        model_path: Path to ONNX model (default: artifacts/model.onnx)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        class_names: List of class names
        use_gpu: Whether to use GPU
    
    Returns:
        YOLOv5ONNX model instance
    """
    global _model
    
    if _model is None:
        # Default path
        if model_path is None:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            model_path = str(project_root / "artifacts" / "model.onnx")
        
        # Default class names for dummy dataset
        if class_names is None:
            class_names = ["class1", "class2", "class3"]
        
        _model = YOLOv5ONNX(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            use_gpu=use_gpu,
        )
    
    return _model


def get_model() -> Optional[YOLOv5ONNX]:
    """Get the loaded model instance."""
    return _model
