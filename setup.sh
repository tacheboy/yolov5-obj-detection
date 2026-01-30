#!/bin/bash
# YOLOv5 Object Detection - Setup Script for Linux/Mac
# This script automates the setup process for the YOLOv5 object detection pipeline

set -e  # Exit on error

echo "========================================"
echo "YOLOv5 Object Detection Pipeline Setup"
echo "========================================"
echo ""

# Check if Python is installed
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"
echo ""

# Check if YOLOv5 is cloned
echo "[2/5] Checking YOLOv5 repository..."
if [ ! -d "yolov5" ]; then
    echo "YOLOv5 not found. Cloning repository..."
    git clone https://github.com/ultralytics/yolov5.git
    echo "YOLOv5 cloned successfully!"
else
    echo "YOLOv5 repository already exists"
fi
echo ""

# Create virtual environment
echo "[3/5] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created!"
else
    echo "Virtual environment already exists"
fi
echo ""

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "[4/5] Installing dependencies..."
echo "This may take several minutes..."
echo ""

echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

echo "Installing PyTorch (CPU version)..."
python -m pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu

echo "Installing additional dependencies..."
python -m pip install matplotlib pandas seaborn pyyaml tqdm opencv-python gitpython psutil scipy thop ultralytics --quiet

echo "All dependencies installed successfully!"
echo ""

# Generate dummy dataset
echo "[5/5] Generating dummy dataset..."
python scripts/create_dummy_dataset.py || {
    echo "Warning: Failed to generate dummy dataset"
    echo "You can run it manually: python scripts/create_dummy_dataset.py"
}
echo ""

# Success message
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Train the model:"
echo "   python scripts/train_model.py --epochs 50 --batch 8"
echo ""
echo "3. Validate the model:"
echo "   python scripts/validate_model.py --verbose"
echo ""
echo "For more information, see README.md"
echo ""
