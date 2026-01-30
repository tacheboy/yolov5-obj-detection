# YOLOv5 Object Detection

Train and validate YOLOv5 models on custom datasets with data augmentation.

## Project Structure

```
yolov5-obj-detection/
├── .venv/                        # Python virtual environment
├── data/
│   ├── dataset.yaml              # Dataset configuration
│   └── dummy/                    # Dummy dataset (generated)
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── scripts/
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

### 1. Clone YOLOv5 (if not already done)

```bash
cd yolov5-obj-detection
git clone https://github.com/ultralytics/yolov5.git
```

### 2. Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install PyTorch (CPU version - faster to install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or for CUDA support (if you have an NVIDIA GPU):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install matplotlib pandas seaborn pyyaml tqdm opencv-python gitpython psutil scipy thop ultralytics
```

### 3. Generate Dummy Dataset

```bash
python scripts/create_dummy_dataset.py
```

This creates a synthetic dataset at `data/dummy/` with:
- 20 training images with colored rectangles
- 5 validation images
- YOLO-format labels (class_id, x_center, y_center, width, height)
- 3 classes: class1, class2, class3

### 4. Train the Model

**Using Python wrapper (recommended):**
```bash
python scripts/train_model.py --epochs 50 --batch 16
```

**Using PowerShell wrapper:**
```powershell
.\scripts\train.ps1 -Epochs 50 -BatchSize 16
```

**Direct YOLOv5 command (from yolov5/ directory):**
```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 50 --data ../data/dataset.yaml --weights yolov5s.pt
```

### 5. Validate the Model

**Using Python wrapper:**
```bash
python scripts/validate_model.py --weights yolov5/runs/train/exp/weights/best.pt
```

**Using PowerShell wrapper:**
```powershell
.\scripts\val.ps1 -Weights "runs/train/exp/weights/best.pt"
```

**Direct YOLOv5 command (from yolov5/ directory):**
```bash
cd yolov5
python val.py --weights runs/train/exp/weights/best.pt --data ../data/dataset.yaml
```

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
python train.py --hyp data/hyps/hyp.no-augmentation.yaml --data ../data/dataset.yaml --weights yolov5s.pt
```

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch` | 16 | Batch size |
| `--img` | 640 | Input image size |
| `--weights` | yolov5s.pt | Pretrained weights (yolov5n/s/m/l/x.pt) |
| `--data` | ../data/dataset.yaml | Dataset configuration |
| `--cache` | None | Cache images (ram/disk) for faster training |
| `--resume` | False | Resume from last checkpoint |

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

### Update dataset.yaml

Edit `data/dataset.yaml` to point to your dataset:
```yaml
path: ../data/your_dataset  # or absolute path
train: images/train
val: images/val
names:
  0: your_class1
  1: your_class2
  # ...
```

## Future: Preprocessing Pipeline

For real datasets, add a preprocessing script at `scripts/preprocess_dataset.py` that:
1. Reads raw data (images + annotations in any format)
2. Converts annotations to YOLO format
3. Outputs to `data/processed/` with the same structure as `data/dummy/`
4. Update `data/dataset.yaml` to point to `../data/processed`

The training and validation pipeline remains unchanged.

## References

- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [Training Custom Data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)
- [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/)
- [Data Augmentation](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/)
