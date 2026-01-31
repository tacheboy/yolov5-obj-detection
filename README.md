# YOLOv5 Object Detection

A complete pipeline for training and validating YOLOv5 models on custom datasets with data augmentation. Includes dummy dataset generation for quick testing and a framework for preprocessing your own datasets.

## Features

- 🚀 Automated dummy dataset generation with synthetic images
- 🔄 YOLOv5 training with built-in data augmentation (mosaic, mixup, flips, etc.)
- 📊 Validation with detailed metrics (mAP50, mAP50-95, precision, recall)
- 💻 Cross-platform wrapper scripts (Python & PowerShell)
- 📝 Easy-to-follow setup for new users
- 🎯 Ready for custom dataset preprocessing
- 🔧 ONNX model export for optimized inference
- 🌐 FastAPI REST service for production deployment
- ⚡ Inference benchmarking with performance reports

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Options](#training-options)
- [Model Variants](#model-variants)
- [Data Augmentation](#data-augmentation)
- [ONNX Export](#onnx-export)
- [FastAPI Inference Service](#fastapi-inference-service)
- [Inference Benchmarking](#inference-benchmarking)
- [Using Your Own Dataset](#using-your-own-dataset)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [References](#references)

## Project Structure

```
yolov5-obj-detection/
├── .venv/                        # Python virtual environment
├── api/
│   ├── app.py                    # FastAPI inference service
│   └── model.py                  # ONNX model loader
├── artifacts/                    # Exported models (generated)
│   ├── best.pt                   # PyTorch weights copy
│   └── model.onnx                # ONNX exported model
├── data/
│   ├── dummy/
│   │   ├── dataset.yaml          # Dummy dataset config
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   ├── fire_data/
│   │   ├── fire_dataset.yaml     # Fire/smoke dataset config
│   │   └── preprocess_fire_dataset.py
├── reports/                      # Performance reports (generated)
│   └── inference_benchmark.md    # Benchmark results
├── scripts/
│   ├── benchmark_inference.py    # Measure inference speed
│   ├── convert_to_onnx.py        # Export model to ONNX
│   ├── create_dummy_dataset.py   # Generate synthetic training data
│   ├── train_model.py            # Python training wrapper
│   ├── validate_model.py         # Python validation wrapper
│   ├── train.ps1                 # PowerShell training wrapper
│   └── val.ps1                   # PowerShell validation wrapper
├── yolov5/                       # YOLOv5 repository (cloned)
├── requirements.txt
└── README.md
```

## Quick Start

Follow these steps to set up and run the complete pipeline:

### Step 1: Clone This Repository

```bash
git clone <your-repo-url>
cd yolov5-obj-detection
```

### Step 2: Clone YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
```

### Step 3: Set Up Python Environment

#### Option A: Automated Setup (Recommended)

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Linux/Mac (Bash):**
```bash
chmod +x setup.sh
./setup.sh
```

#### Option B: Manual Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install PyTorch CPU version (or see GPU instructions below)
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install matplotlib pandas seaborn pyyaml tqdm opencv-python gitpython psutil scipy thop ultralytics
```

**For GPU Support (NVIDIA CUDA):**
```bash
# Instead of CPU PyTorch, install CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Generate Dummy Dataset

```bash
# Activate venv if not already active
# Windows: .\.venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

python scripts/create_dummy_dataset.py
```

**Output:**
- `data/dummy/images/train/` - 20 training images with colored rectangles
- `data/dummy/images/val/` - 5 validation images
- `data/dummy/labels/` - YOLO-format labels (class_id, x_center, y_center, width, height)
- 3 classes: class1, class2, class3
- Dataset config: `data/dummy/dataset.yaml`

**Customize dataset generation:**
```bash
python scripts/create_dummy_dataset.py --num_train 50 --num_val 10 --img_size 640
```

### Step 5: Train the Model

**Using Python wrapper (recommended):**
```bash
python scripts/train_model.py --epochs 50 --batch 8 --cache ram
```

GPU is used automatically if available. To force a device:
```bash
python scripts/train_model.py --device cpu
python scripts/train_model.py --device 0
```

**Using PowerShell wrapper (Windows):**
```powershell
.\scripts\train.ps1 -Epochs 50 -BatchSize 8
```

**Direct command (from yolov5/ directory):**
```bash
cd yolov5
..\.venv\Scripts\python.exe train.py --img 640 --batch 8 --epochs 50 --data ../data/dummy/dataset.yaml --weights yolov5s.pt
```

**Training outputs:**
- Weights: `yolov5/runs/train/exp/weights/best.pt` and `last.pt`
- Metrics: `yolov5/runs/train/exp/results.csv`
- Visualizations: `yolov5/runs/train/exp/*.png` (loss curves, PR curves, etc.)

### Step 6: Validate the Model

**Using Python wrapper:**
```bash
python scripts/validate_model.py --weights yolov5/runs/train/exp/weights/best.pt --verbose
```

GPU is used automatically if available. To force a device:
```bash
python scripts/validate_model.py --device cpu
python scripts/validate_model.py --device 0
```

**Using PowerShell wrapper:**
```powershell
.\scripts\val.ps1 -Weights "runs/train/exp/weights/best.pt" -Verbose
```

**Direct command (from yolov5/ directory):**
```bash
cd yolov5
..\.venv\Scripts\python.exe val.py --weights runs/train/exp/weights/best.pt --data ../data/dummy/dataset.yaml --verbose
```

**Validation outputs:**
- Metrics: mAP50, mAP50-95, precision, recall per class
- Visualizations: `yolov5/runs/val/exp/*.png` (confusion matrix, PR curves, etc.)

## Data Augmentation

YOLOv5 includes built-in data augmentation during training:
- **Mosaic augmentation**: Combines 4 images into one
- **MixUp**: Blends two images together
- **Horizontal flip**
- **Scale, translation, rotation**
- **HSV color space augmentation**
- **Copy-paste augmentation** (for segmentation)

Augmentation is **enabled by default**. To disable:
```bash
python scripts/train_model.py --no-augment
```

Or use a custom hyperparameter file:
```bash
cd yolov5
python train.py --hyp data/hyps/hyp.no-augmentation.yaml --data ../data/dummy/dataset.yaml --weights yolov5s.pt
```

## ONNX Export

Export your trained model to ONNX format for optimized inference:

```bash
# Auto-detect latest trained model and export
python scripts/convert_to_onnx.py

# Specify weights and output path
python scripts/convert_to_onnx.py --weights yolov5/runs/train/exp/weights/best.pt --output artifacts/model.onnx

# Export with dynamic batch size
python scripts/convert_to_onnx.py --dynamic --simplify
```

**Export Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--weights` | auto-detect | Path to PyTorch weights |
| `--output` | artifacts/model.onnx | Output ONNX path |
| `--img` | 640 | Input image size |
| `--opset` | 12 | ONNX opset version |
| `--simplify` | False | Simplify model with onnxslim |
| `--dynamic` | False | Enable dynamic batch size |

**Output:**
- `artifacts/model.onnx` - Exported ONNX model
- `artifacts/best.pt` - Copy of source PyTorch weights

## FastAPI Inference Service

Run object detection as a REST API:

### Start the Server

```bash
# Install API dependencies (if not done)
pip install fastapi uvicorn python-multipart onnxruntime

# Start the server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Or run directly
python api/app.py
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/model` | GET | Model information |
| `/predict` | POST | Run inference on image |
| `/docs` | GET | Swagger UI documentation |

### Example: Run Inference

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

**Using Python requests:**
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"conf_threshold": 0.5}
    )
    print(response.json())
```

**Response format:**
```json
{
  "success": true,
  "num_detections": 2,
  "detections": [
    {
      "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 150},
      "confidence": 0.92,
      "class_id": 0,
      "class_name": "circle"
    }
  ],
  "image_size": {"width": 640, "height": 480}
}
```

### Configuration

Set environment variables to customize:

```bash
# Custom model path
export MODEL_PATH=/path/to/model.onnx

# Adjust thresholds
export CONF_THRESHOLD=0.3
export IOU_THRESHOLD=0.5

# Disable GPU (CPU fallback is automatic if CUDA is unavailable)
export USE_GPU=false

# Custom class names
export CLASS_NAMES=cat,dog,bird
```

## Inference Benchmarking

Measure inference speed for both PyTorch and ONNX models:

```bash
# Run benchmark with defaults
python scripts/benchmark_inference.py

# Custom settings
python scripts/benchmark_inference.py --runs 100 --warmup 20 --img 640

# Force CPU or GPU
python scripts/benchmark_inference.py --device cpu
python scripts/benchmark_inference.py --device cuda

# Specify model paths
python scripts/benchmark_inference.py \
  --pt-weights artifacts/best.pt \
  --onnx-weights artifacts/model.onnx
```

Note: ONNX GPU inference requires `onnxruntime-gpu`. If it's not installed, the benchmark falls back to CPU.

**Benchmark Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--pt-weights` | auto-detect | PyTorch model path |
| `--onnx-weights` | artifacts/model.onnx | ONNX model path |
| `--img` | 640 | Input image size |
| `--runs` | 50 | Number of benchmark runs |
| `--warmup` | 10 | Warmup runs before timing |
| `--output` | reports/inference_benchmark.md | Report output path |
| `--device` | auto | Device for PyTorch/ONNX (auto/cpu/cuda) |

**Output:**
- `reports/inference_benchmark.md` - Detailed performance report with latency statistics and FPS

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--img` | 640 | Input image size |
| `--weights` | yolov5s.pt | Pretrained weights (yolov5n/s/m/l/x.pt) |
| `--data` | ../data/dummy/dataset.yaml | Dataset configuration |
| `--cache` | None | Cache images (ram/disk) for faster training |
| `--resume` | False | Resume from last checkpoint |
| `--device` | auto | Device for training (auto/cpu/cuda or index) |

## Model Variants

| Model | Size | mAP | Speed | Use Case |
|-------|------|-----|-------|----------|
| YOLOv5n | Nano | Lower | Fastest | Edge devices, mobile |
| YOLOv5s | Small | Good | Fast | General purpose |
| YOLOv5m | Medium | Better | Medium | Balance of speed/accuracy |
| YOLOv5l | Large | High | Slower | High accuracy needed |
| YOLOv5x | XLarge | Highest | Slowest | Maximum accuracy |

## Using Your Own Dataset

### Dataset Format

YOLOv5 expects this directory structure:
```
your_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

### Label Format

Each `.txt` file contains one line per object:
```
class_id x_center y_center width height
```
- All values are normalized to [0, 1]
- `x_center`, `y_center`: Center of bounding box
- `width`, `height`: Size of bounding box
- `class_id`: Zero-indexed class number

Example:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.1 0.15
```

### Create a dataset YAML

Create a dataset YAML under `data/` and point it to your dataset:
```yaml
path: ../data/your_dataset  # or absolute path
train: images/train
val: images/val
names:
  0: your_class1
  1: your_class2
  # ...
```

## Fire Dataset Preprocessing (Smoke/Fire)

If you have the fire/smoke dataset described in this repo:

```bash
python data/fire_data/preprocess_fire_dataset.py --dataset-root "/kaggle/working/D Fire Dataset" --create-empty-labels
```

This will output a YOLOv5-ready dataset under `data/fire_data/`.
Train with:
```bash
python scripts/train_model.py --data ../data/fire_data/fire_dataset.yaml
```

## Troubleshooting

### PyTorch Installation Issues

**Windows Long Path Error:**
If you encounter "path too long" errors, use the virtual environment approach (already done in setup):
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA Version Mismatch:**
Check your CUDA version and install the matching PyTorch:
```bash
nvidia-smi  # Check CUDA version
# Then install matching PyTorch from https://pytorch.org/
```

### Training Issues

**Out of Memory:**
- Reduce batch size: `--batch 4` or `--batch 2`
- Use smaller model: `--weights yolov5n.pt` (nano) instead of yolov5s.pt

**Slow Training:**
- Use `--cache ram` to cache images in RAM
- Reduce image size: `--img 416` instead of 640
- Use GPU instead of CPU (see GPU setup in Quick Start)

### Import Errors

If you get "Module not found" errors:
```bash
# Make sure virtual environment is activated
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -r yolov5/requirements.txt
```

## Advanced Usage

### Custom Training Parameters

```bash
# High accuracy (slower)
python scripts/train_model.py --epochs 100 --batch 16 --weights yolov5l.pt

# Fast training (lower accuracy)
python scripts/train_model.py --epochs 50 --batch 32 --weights yolov5n.pt --img 416

# Resume from checkpoint
python scripts/train_model.py --resume

# Disable augmentation
python scripts/train_model.py --no-augment
```

### Hyperparameter Tuning

Edit or create custom hyperparameter files in `yolov5/data/hyps/`:
```bash
cd yolov5
python train.py --hyp data/hyps/hyp.scratch-high.yaml --data ../data/dummy/dataset.yaml
```

### Export Trained Model

Export to different formats for deployment:
```bash
cd yolov5
python export.py --weights runs/train/exp/weights/best.pt --include onnx tflite torchscript
```

## Project Maintenance

### Updating YOLOv5

```bash
cd yolov5
git pull origin master
cd ..
```

### Cleaning Up

```bash
# Remove training outputs
rm -rf yolov5/runs/

# Remove dummy dataset
rm -rf data/dummy/

# Regenerate dummy dataset
python scripts/create_dummy_dataset.py
```

## Contributing

When contributing to this repository:
1. Fork the repository
2. Create a feature branch
3. Test your changes with the dummy dataset
4. Update documentation if needed
5. Submit a pull request

## License

This project uses YOLOv5 which is licensed under AGPL-3.0. See [YOLOv5 License](https://github.com/ultralytics/yolov5/blob/master/LICENSE) for details.

## References

- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [Training Custom Data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)
- [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/)
- [Data Augmentation](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
- [YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/)
