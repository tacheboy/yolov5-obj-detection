"""
YOLOv5 Inference Benchmark Script

Measures inference speed for both PyTorch and ONNX models.
Run from project root: python scripts/benchmark_inference.py

Usage:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --runs 100 --warmup 10
    python scripts/benchmark_inference.py --pt-weights artifacts/best.pt --onnx-weights artifacts/model.onnx
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Check for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def get_project_paths():
    """Get project paths."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    artifacts_dir = project_root / "artifacts"
    reports_dir = project_root / "reports"
    yolo_dir = project_root / "yolov5"
    return project_root, artifacts_dir, reports_dir, yolo_dir


def find_best_weights(yolo_dir: Path) -> str:
    """Try to find the best weights from the most recent training run."""
    runs_dir = yolo_dir / "runs" / "train"
    if not runs_dir.exists():
        return None
    
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


def create_dummy_image(img_size: int = 640) -> np.ndarray:
    """Create a dummy image for benchmarking."""
    return np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)


def preprocess_image(img: np.ndarray, img_size: int = 640) -> np.ndarray:
    """Preprocess image for model input."""
    # Resize
    img = cv2.resize(img, (img_size, img_size))
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # HWC to CHW
    img = img.transpose(2, 0, 1)
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def normalize_device(device: str) -> str:
    """Normalize device input for PyTorch."""
    device = device.lower()
    if device == "cpu":
        return "cpu"
    if device == "auto":
        return "cuda" if CUDA_AVAILABLE else "cpu"
    if device.startswith("cuda"):
        return device
    if device.isdigit() or "," in device:
        return "cuda"
    if device in {"gpu"}:
        return "cuda"
    return device

def select_onnx_providers(use_gpu: bool) -> list:
    """Select ONNX Runtime execution providers."""
    if not ONNX_AVAILABLE:
        return []
    available = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def benchmark_pytorch(weights_path: str, img: np.ndarray, runs: int, warmup: int, device: str):
    """Benchmark PyTorch model inference."""
    if not TORCH_AVAILABLE:
        return None, "PyTorch not available"
    
    try:
        # Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        model.to(device)
        model.eval()
        
        # Create tensor input
        img_tensor = torch.from_numpy(img).to(device)
        
        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(img_tensor)
        
        # Synchronize if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"  Benchmarking ({runs} runs)...")
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                _ = model(img_tensor)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        return times, None
    except Exception as e:
        return None, str(e)


def benchmark_onnx(onnx_path: str, img: np.ndarray, runs: int, warmup: int, use_gpu: bool):
    """Benchmark ONNX model inference."""
    if not ONNX_AVAILABLE:
        return None, "ONNX Runtime not available"
    
    try:
        # Select execution provider
        providers = select_onnx_providers(use_gpu)
        if not providers:
            return None, "ONNX Runtime not available"
        
        # Load model
        session = ort.InferenceSession(onnx_path, providers=providers)
        input_name = session.get_inputs()[0].name
        
        # Check which provider is actually being used
        actual_provider = session.get_providers()[0]
        
        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            _ = session.run(None, {input_name: img})
        
        # Benchmark
        print(f"  Benchmarking ({runs} runs)...")
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = session.run(None, {input_name: img})
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        return times, actual_provider
    except Exception as e:
        return None, str(e)


def calculate_stats(times: list) -> dict:
    """Calculate statistics from timing results."""
    if not times:
        return {}
    
    times = np.array(times)
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
        "fps": 1000.0 / np.mean(times),
    }


def generate_report(results: dict, output_path: Path, img_size: int, runs: int):
    """Generate markdown performance report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# YOLOv5 Inference Benchmark Report

**Generated:** {timestamp}  
**Image Size:** {img_size}x{img_size}  
**Benchmark Runs:** {runs}  

## System Information

- **PyTorch Available:** {TORCH_AVAILABLE}
- **CUDA Available:** {CUDA_AVAILABLE}
- **ONNX Runtime Available:** {ONNX_AVAILABLE}

## Results Summary

| Model | Device | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | P95 (ms) | FPS |
|-------|--------|-----------|----------|----------|----------|----------|-----|
"""
    
    for name, data in results.items():
        if data.get("stats"):
            stats = data["stats"]
            device = data.get("device", "N/A")
            report += f"| {name} | {device} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['p95']:.2f} | {stats['fps']:.1f} |\n"
        elif data.get("error"):
            report += f"| {name} | N/A | Error: {data['error']} | | | | | |\n"
    
    report += """
## Detailed Results

"""
    
    for name, data in results.items():
        report += f"### {name}\n\n"
        if data.get("stats"):
            stats = data["stats"]
            report += f"""- **Device:** {data.get('device', 'N/A')}
- **Mean Latency:** {stats['mean']:.2f} ms
- **Standard Deviation:** {stats['std']:.2f} ms
- **Min Latency:** {stats['min']:.2f} ms
- **Max Latency:** {stats['max']:.2f} ms
- **Median Latency:** {stats['median']:.2f} ms
- **95th Percentile:** {stats['p95']:.2f} ms
- **99th Percentile:** {stats['p99']:.2f} ms
- **Throughput:** {stats['fps']:.1f} FPS

"""
        elif data.get("error"):
            report += f"**Error:** {data['error']}\n\n"
    
    report += """## Notes

- All times are in milliseconds (ms)
- FPS = 1000 / mean_latency_ms
- P95/P99 = 95th/99th percentile latency
- ONNX Runtime may use different execution providers (CUDA, CPU)
- PyTorch benchmarks include model forward pass only
"""
    
    with open(output_path, "w") as f:
        f.write(report)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv5 inference speed"
    )
    parser.add_argument(
        "--pt-weights",
        type=str,
        default=None,
        help="Path to PyTorch weights (default: artifacts/best.pt or auto-detect)",
    )
    parser.add_argument(
        "--onnx-weights",
        type=str,
        default=None,
        help="Path to ONNX model (default: artifacts/model.onnx)",
    )
    parser.add_argument(
        "--img", type=int, default=640, help="Image size (default: 640)"
    )
    parser.add_argument(
        "--runs", type=int, default=50, help="Number of benchmark runs (default: 50)"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup runs (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output report path (default: reports/inference_benchmark.md)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for PyTorch/ONNX: auto, cpu, cuda (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Get paths
    project_root, artifacts_dir, reports_dir, yolo_dir = get_project_paths()
    
    # Find weights
    pt_weights = args.pt_weights
    if pt_weights is None:
        # Try artifacts first, then auto-detect
        if (artifacts_dir / "best.pt").exists():
            pt_weights = str(artifacts_dir / "best.pt")
        else:
            pt_weights = find_best_weights(yolo_dir)
    
    onnx_weights = args.onnx_weights
    if onnx_weights is None:
        onnx_weights = str(artifacts_dir / "model.onnx")
    
    output_path = Path(args.output) if args.output else reports_dir / "inference_benchmark.md"
    
    print("=" * 60)
    print("YOLOv5 Inference Benchmark")
    print("=" * 60)
    print(f"PyTorch weights: {pt_weights}")
    print(f"ONNX weights: {onnx_weights}")
    print(f"Image size: {args.img}x{args.img}")
    print(f"Benchmark runs: {args.runs}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Output report: {output_path}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print("=" * 60)
    
    # Create dummy image
    print("\nCreating test image...")
    dummy_img = create_dummy_image(args.img)
    preprocessed_img = preprocess_image(dummy_img, args.img)
    
    results = {}
    
    # Determine device
    device = normalize_device(args.device)
    if device.startswith("cuda") and not CUDA_AVAILABLE:
        device = "cpu"
    
    # Benchmark PyTorch
    if pt_weights and Path(pt_weights).exists():
        print(f"\n[1/2] Benchmarking PyTorch model on {device.upper()}...")
        times, error = benchmark_pytorch(pt_weights, preprocessed_img, args.runs, args.warmup, device)
        if times:
            results["PyTorch"] = {
                "stats": calculate_stats(times),
                "device": device.upper(),
            }
        else:
            results["PyTorch"] = {"error": error}
    else:
        print("\n[1/2] Skipping PyTorch benchmark (weights not found)")
        results["PyTorch"] = {"error": "Weights not found"}
    
    # Benchmark ONNX
    if onnx_weights and Path(onnx_weights).exists():
        print(f"\n[2/2] Benchmarking ONNX model...")
        use_gpu = device.startswith("cuda")
        times, provider_or_error = benchmark_onnx(
            onnx_weights, preprocessed_img, args.runs, args.warmup, use_gpu=use_gpu
        )
        if times:
            results["ONNX"] = {
                "stats": calculate_stats(times),
                "device": provider_or_error,
            }
        else:
            results["ONNX"] = {"error": provider_or_error}
    else:
        print("\n[2/2] Skipping ONNX benchmark (model not found)")
        results["ONNX"] = {"error": "Model not found - run convert_to_onnx.py first"}
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(results, output_path, args.img, args.runs)
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print(f"Report saved to: {output_path}")
    print("=" * 60)
    
    # Print summary
    print("\nSummary:")
    for name, data in results.items():
        if data.get("stats"):
            stats = data["stats"]
            print(f"  {name} ({data.get('device', 'N/A')}): {stats['mean']:.2f} ms ({stats['fps']:.1f} FPS)")
        elif data.get("error"):
            print(f"  {name}: Error - {data['error']}")


if __name__ == "__main__":
    main()
