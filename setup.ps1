# YOLOv5 Object Detection - Setup Script for Windows
# This script automates the setup process for the YOLOv5 object detection pipeline
# scr   
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "YOLOv5 Object Detection Pipeline Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}

$pythonVersion = python --version
Write-Host "Found: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check if YOLOv5 is cloned
Write-Host "[2/5] Checking YOLOv5 repository..." -ForegroundColor Yellow
if (-not (Test-Path "yolov5")) {
    Write-Host "YOLOv5 not found. Cloning repository..." -ForegroundColor Yellow
    git clone https://github.com/ultralytics/yolov5.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to clone YOLOv5 repository" -ForegroundColor Red
        Write-Host "Please ensure git is installed and you have internet access" -ForegroundColor Red
        exit 1
    }
    Write-Host "YOLOv5 cloned successfully!" -ForegroundColor Green
} else {
    Write-Host "YOLOv5 repository already exists" -ForegroundColor Green
}
Write-Host ""

# Create virtual environment
Write-Host "[3/5] Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created!" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}
Write-Host ""

# Install dependencies
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Yellow
Write-Host ""

# Activate venv and install PyTorch
Write-Host "Installing PyTorch (CPU version)..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
& .\.venv\Scripts\python.exe -m pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

Write-Host "Installing additional dependencies..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install matplotlib pandas seaborn pyyaml tqdm opencv-python gitpython psutil scipy thop ultralytics --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host "All dependencies installed successfully!" -ForegroundColor Green
Write-Host ""

# Generate dummy dataset
Write-Host "[5/5] Generating dummy dataset..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe scripts\create_dummy_dataset.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Failed to generate dummy dataset" -ForegroundColor Yellow
    Write-Host "You can run it manually: python scripts/create_dummy_dataset.py" -ForegroundColor Yellow
} else {
    Write-Host "Dummy dataset generated successfully!" -ForegroundColor Green
}
Write-Host ""

# Success message
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Activate the virtual environment:" -ForegroundColor White
Write-Host "   .\.venv\Scripts\activate" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Train the model:" -ForegroundColor White
Write-Host "   python scripts/train_model.py --epochs 50 --batch 8" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Validate the model:" -ForegroundColor White
Write-Host "   python scripts/validate_model.py --verbose" -ForegroundColor Gray
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Cyan
Write-Host ""
