# Setup Guide for YOLOv5 Object Detection

This guide provides detailed setup instructions for different scenarios and operating systems.

## Prerequisites

### Required Software

1. **Python 3.8 or higher**
   - Windows: Download from [python.org](https://www.python.org/)
   - Linux: `sudo apt install python3 python3-pip python3-venv`
   - Mac: `brew install python3`

2. **Git**
   - Windows: Download from [git-scm.com](https://git-scm.com/)
   - Linux: `sudo apt install git`
   - Mac: `brew install git`

3. **CUDA Toolkit (Optional, for GPU training)**
   - Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
   - Check compatibility with your GPU

### System Requirements

**Minimum:**
- RAM: 8GB
- Storage: 5GB free space
- CPU: Multi-core processor

**Recommended:**
- RAM: 16GB+
- Storage: 20GB+ free space
- GPU: NVIDIA GPU with 6GB+ VRAM
- CUDA: 11.8 or higher

## Automated Setup

### Windows

1. Open PowerShell in the project directory
2. Run the setup script:
   ```powershell
   .\setup.ps1
   ```
3. If you encounter execution policy issues:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\setup.ps1
   ```

### Linux/Mac

1. Open Terminal in the project directory
2. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
3. Run the setup script:
   ```bash
   ./setup.sh
   ```

## Manual Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd yolov5-obj-detection
```

### 2. Clone YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
```

### 3. Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

**For CPU-only (faster setup):**
```bash
pip install --upgrade pip
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib pandas seaborn pyyaml tqdm opencv-python gitpython psutil scipy thop ultralytics
```

**For CUDA 11.8 (NVIDIA GPU):**
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib pandas seaborn pyyaml tqdm opencv-python gitpython psutil scipy thop ultralytics
```

**For CUDA 12.1:**
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib pandas seaborn pyyaml tqdm opencv-python gitpython psutil scipy thop ultralytics
```

### 5. Generate Dummy Dataset

```bash
python scripts/create_dummy_dataset.py
```

## Verification

Test that everything is installed correctly:

```bash
# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check CUDA availability (if GPU setup)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify dummy dataset was created
# Windows:
dir data\dummy
# Linux/Mac:
ls -la data/dummy
```

You should see:
- PyTorch version printed
- CUDA status (True if GPU, False if CPU)
- Dummy dataset folders: `images/` and `labels/` with train/val subfolders

## Next Steps

After successful setup:

1. **Quick test (3 epochs):**
   ```bash
   python scripts/train_model.py --epochs 3 --batch 4
   ```

2. **Full training:**
   ```bash
   python scripts/train_model.py --epochs 50 --batch 8 --cache ram
   ```

3. **Validation:**
   ```bash
   python scripts/validate_model.py --verbose
   ```

## Common Issues

### Issue: "command not found: python"

**Solution:**
- Windows: Use `python` or add Python to PATH
- Linux/Mac: Use `python3` instead of `python`

### Issue: "No module named 'torch'"

**Solution:**
```bash
# Make sure virtual environment is activated
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Reinstall PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Long path errors on Windows

**Solution:**
Enable long paths in Windows:
1. Open Registry Editor (regedit)
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to 1
4. Restart computer

Or use the virtual environment (which has shorter paths):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Issue: Out of memory during training

**Solution:**
- Reduce batch size: `--batch 2` or `--batch 4`
- Reduce image size: `--img 416`
- Use smaller model: `--weights yolov5n.pt`
- Close other applications

### Issue: Git clone fails

**Solution:**
```bash
# Check git installation
git --version

# Try with HTTPS
git clone https://github.com/ultralytics/yolov5.git

# Or manually download and extract from:
# https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip
```

## Updating the Environment

To update dependencies:

```bash
# Activate virtual environment
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Update pip
pip install --upgrade pip

# Update all packages
pip install --upgrade torch torchvision matplotlib pandas seaborn pyyaml tqdm opencv-python gitpython psutil scipy thop ultralytics
```

## Uninstalling

To completely remove the environment:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment folder
# Windows:
rmdir /s .venv
# Linux/Mac:
rm -rf .venv

# Remove YOLOv5
# Windows:
rmdir /s yolov5
# Linux/Mac:
rm -rf yolov5

# Remove generated data
# Windows:
rmdir /s data\dummy
# Linux/Mac:
rm -rf data/dummy
```

## Getting Help

If you encounter issues:

1. Check the [main README](README.md) for usage instructions
2. Review this setup guide thoroughly
3. Check [YOLOv5 documentation](https://docs.ultralytics.com/yolov5/)
4. Open an issue on GitHub with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce the issue
