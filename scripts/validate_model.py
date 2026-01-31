"""
YOLOv5 Validation Wrapper Script

Validates a trained YOLOv5 model on the validation dataset.
Run from project root: python scripts/validate_model.py --weights path/to/best.pt

Usage:
    python scripts/validate_model.py --weights yolov5/runs/train/exp/weights/best.pt
    python scripts/validate_model.py --weights yolov5/runs/train/exp/weights/best.pt --verbose
"""

import argparse
import os
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
    
    val_py = yolo_dir / "val.py"
    if not val_py.exists():
        print(f"Error: val.py not found at {val_py}")
        sys.exit(1)


def find_best_weights(yolo_dir: Path) -> str:
    """Try to find the best weights from the most recent training run."""
    runs_dir = yolo_dir / "runs" / "train"
    if not runs_dir.exists():
        return None
    
    # Find most recent experiment directory
    exp_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    
    for exp_dir in exp_dirs:
        best_pt = exp_dir / "weights" / "best.pt"
        if best_pt.exists():
            return str(best_pt)
    
    return None

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
        description="Validate YOLOv5 model on custom dataset"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: auto-detect from latest training run)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset YAML path (default: ../data/dummy/dataset.yaml relative to yolov5/)",
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--img", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Task: val, test, or train (default: val)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.001,
        help="Confidence threshold (default: 0.001)",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.6,
        help="NMS IoU threshold (default: 0.6)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Report mAP by class"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="Save results to *.txt"
    )
    parser.add_argument(
        "--save-json", action="store_true", help="Save results to COCO JSON"
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
    
    # Find or validate weights path
    weights_path = args.weights
    if weights_path is None:
        weights_path = find_best_weights(yolo_dir)
        if weights_path is None:
            print("Error: No weights specified and no trained model found.")
            print("Please train a model first or specify --weights path/to/model.pt")
            sys.exit(1)
        print(f"Auto-detected weights: {weights_path}")
    
    # Default data path (relative to yolov5 directory)
    data_path = args.data or "../data/dummy/dataset.yaml"

    # Resolve device
    device_arg = args.device
    if args.device.lower() == "auto":
        cuda_available = detect_cuda_available(python_exe)
        device_arg = "0" if cuda_available else "cpu"
    
    # Build validation command
    cmd = [
        python_exe,
        "val.py",
        "--weights", weights_path,
        "--data", data_path,
        "--batch-size", str(args.batch),
        "--img", str(args.img),
        "--task", args.task,
        "--conf-thres", str(args.conf_thres),
        "--iou-thres", str(args.iou_thres),
        "--device", device_arg,
    ]
    
    if args.verbose:
        cmd.append("--verbose")
    
    if args.save_txt:
        cmd.append("--save-txt")
    
    if args.save_json:
        cmd.append("--save-json")
    
    print("=" * 60)
    print("YOLOv5 Validation")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"YOLOv5 directory: {yolo_dir}")
    print(f"Weights: {weights_path}")
    print(f"Data config: {data_path}")
    print(f"Task: {args.task}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Device: {device_arg}")
    print("=" * 60)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Change to yolov5 directory and run validation
    original_dir = os.getcwd()
    os.chdir(yolo_dir)
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("Validation complete!")
            print(f"Results saved to: {yolo_dir / 'runs' / 'val'}")
            print("=" * 60)
        else:
            print(f"\nValidation failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
