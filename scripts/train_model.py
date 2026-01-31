"""
YOLOv5 Training Wrapper Script

Trains a YOLOv5 model on the custom dataset with data augmentation.
Run from project root: python scripts/train_model.py

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 100 --batch 8 --weights yolov5m.pt
"""

import os

# --- SILENCE WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_MODE'] = 'disabled'

import argparse
import subprocess
import sys
from pathlib import Path


def get_project_paths():
    """Get project and YOLOv5 paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    yolo_dir = project_root / "yolov5"
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    return project_root, yolo_dir, venv_python


def check_yolov5_installed(yolo_dir: Path):
    """Check if YOLOv5 is installed."""
    if not yolo_dir.exists():
        print(f"Error: YOLOv5 not found at {yolo_dir}")
        print("Please clone YOLOv5 first:")
        print("  git clone https://github.com/ultralytics/yolov5.git")
        sys.exit(1)
    
    train_py = yolo_dir / "train.py"
    if not train_py.exists():
        print(f"Error: train.py not found at {train_py}")
        sys.exit(1)


def install_yolov5_requirements(yolo_dir: Path):
    """Install YOLOv5 requirements if needed."""
    requirements_file = yolo_dir / "requirements.txt"
    if requirements_file.exists():
        print("Installing YOLOv5 requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=False,
        )

def detect_cuda_available(python_exe: str) -> bool:
    """Detect CUDA availability using the selected Python interpreter."""
    try:
        result = subprocess.run(
            [python_exe, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        return result.stdout.strip().lower() == "true"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv5 model on custom dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--img", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov5s.pt",
        help="Pretrained weights (default: yolov5s.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset YAML path (default: ../data/dummy/dataset.yaml relative to yolov5/)",
    )
    parser.add_argument(
        "--name", type=str, default="", help="Experiment name (optional)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (use hyp.no-augmentation.yaml)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        choices=["ram", "disk"],
        default=None,
        help="Cache images for faster training",
    )
    parser.add_argument(
        "--install-requirements",
        action="store_true",
        help="Install YOLOv5 requirements before training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, or CUDA device index (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Get paths
    project_root, yolo_dir, venv_python = get_project_paths()
    check_yolov5_installed(yolo_dir)
    
    # Determine Python interpreter (prefer venv if available)
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    
    # Optionally install requirements
    if args.install_requirements:
        install_yolov5_requirements(yolo_dir)
    
    # Default data path (relative to yolov5 directory)
    data_path = args.data or "../data/dummy/dataset.yaml"

    # Resolve device
    device_arg = args.device
    if args.device.lower() == "auto":
        cuda_available = detect_cuda_available(python_exe)
        device_arg = "0" if cuda_available else "cpu"
    
    # Build training command
    cmd = [
        python_exe,
        "train.py",
        "--img", str(args.img),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", data_path,
        "--weights", args.weights,
        "--device", device_arg,
    ]
    
    if args.name:
        cmd.extend(["--name", args.name])
    
    if args.resume:
        cmd.append("--resume")
    
    if args.no_augment:
        cmd.extend(["--hyp", "data/hyps/hyp.no-augmentation.yaml"])
    
    if args.cache:
        cmd.extend(["--cache", args.cache])
    
    print("=" * 60)
    print("YOLOv5 Training")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"YOLOv5 directory: {yolo_dir}")
    print(f"Data config: {data_path}")
    print(f"Weights: {args.weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Device: {device_arg}")
    print(f"Augmentation: {'disabled' if args.no_augment else 'enabled (default)'}")
    print("=" * 60)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Change to yolov5 directory and run training
    original_dir = os.getcwd()
    os.chdir(yolo_dir)
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("Training complete!")
            print(f"Results saved to: {yolo_dir / 'runs' / 'train'}")
            print("=" * 60)
        else:
            print(f"\nTraining failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
