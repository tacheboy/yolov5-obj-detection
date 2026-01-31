"""
YOLOv5 ONNX Export Script

Converts a trained YOLOv5 PyTorch model (.pt) to ONNX format.
Run from project root: python scripts/convert_to_onnx.py

Usage:
    python scripts/convert_to_onnx.py
    python scripts/convert_to_onnx.py --weights yolov5/runs/train/exp/weights/best.pt
    python scripts/convert_to_onnx.py --output artifacts/model.onnx --img 640
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_project_paths():
    """Get project and YOLOv5 paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    yolo_dir = project_root / "yolov5"
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    artifacts_dir = project_root / "artifacts"
    return project_root, yolo_dir, venv_python, artifacts_dir


def check_yolov5_installed(yolo_dir: Path):
    """Check if YOLOv5 is installed."""
    if not yolo_dir.exists():
        print(f"Error: YOLOv5 not found at {yolo_dir}")
        print("Please clone YOLOv5 first:")
        print("  git clone https://github.com/ultralytics/yolov5.git")
        sys.exit(1)
    
    export_py = yolo_dir / "export.py"
    if not export_py.exists():
        print(f"Error: export.py not found at {export_py}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv5 model to ONNX format"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: auto-detect from latest training run)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX path (default: artifacts/model.onnx)",
    )
    parser.add_argument(
        "--img", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--opset", type=int, default=12, help="ONNX opset version (default: 12)"
    )
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify ONNX model with onnxslim"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable dynamic axes for batch size"
    )
    
    args = parser.parse_args()
    
    # Get paths
    project_root, yolo_dir, venv_python, artifacts_dir = get_project_paths()
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
    
    # Ensure weights path exists
    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}")
        sys.exit(1)
    
    # Create artifacts directory
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output path
    output_path = args.output or str(artifacts_dir / "model.onnx")
    
    # Build export command
    cmd = [
        python_exe,
        "export.py",
        "--weights", str(weights_path),
        "--img", str(args.img),
        "--include", "onnx",
        "--opset", str(args.opset),
    ]
    
    if args.simplify:
        cmd.append("--simplify")
    
    if args.dynamic:
        cmd.append("--dynamic")
    
    print("=" * 60)
    print("YOLOv5 ONNX Export")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"YOLOv5 directory: {yolo_dir}")
    print(f"Input weights: {weights_path}")
    print(f"Output ONNX: {output_path}")
    print(f"Image size: {args.img}")
    print(f"ONNX opset: {args.opset}")
    print("=" * 60)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Change to yolov5 directory and run export
    original_dir = os.getcwd()
    os.chdir(yolo_dir)
    
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            # YOLOv5 export.py creates .onnx next to the .pt file
            # Move it to artifacts directory
            exported_onnx = weights_path.with_suffix(".onnx")
            
            if exported_onnx.exists():
                # Move to artifacts directory
                final_path = Path(output_path)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(exported_onnx), str(final_path))
                
                print("\n" + "=" * 60)
                print("Export complete!")
                print(f"ONNX model saved to: {final_path}")
                print(f"Model size: {final_path.stat().st_size / (1024*1024):.2f} MB")
                print("=" * 60)
                
                # Also copy the source .pt for reference
                pt_copy = artifacts_dir / "best.pt"
                if not pt_copy.exists():
                    shutil.copy(str(weights_path), str(pt_copy))
                    print(f"PyTorch weights copied to: {pt_copy}")
            else:
                print(f"\nWarning: Expected ONNX file not found at {exported_onnx}")
                print("Check the export output above for errors.")
                sys.exit(1)
        else:
            print(f"\nExport failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
