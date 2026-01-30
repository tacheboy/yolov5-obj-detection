"""
Create a dummy dataset for YOLOv5 training and validation.

Generates synthetic images with colored rectangles and corresponding YOLO-format labels.
The dataset structure follows YOLOv5's expected layout:
    data/dummy/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Usage:
    python scripts/create_dummy_dataset.py
    python scripts/create_dummy_dataset.py --num_train 50 --num_val 10 --img_size 640
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Default configuration
DEFAULT_NUM_TRAIN = 20
DEFAULT_NUM_VAL = 5
DEFAULT_IMG_SIZE = 640
DEFAULT_NUM_CLASSES = 3
DEFAULT_CLASS_NAMES = ["class1", "class2", "class3"]

# Colors for each class (RGB)
CLASS_COLORS = [
    (255, 0, 0),    # Red for class1
    (0, 255, 0),    # Green for class2
    (0, 0, 255),    # Blue for class3
]


def create_directories(base_path: Path) -> dict:
    """Create the dataset directory structure."""
    dirs = {
        "train_images": base_path / "images" / "train",
        "val_images": base_path / "images" / "val",
        "train_labels": base_path / "labels" / "train",
        "val_labels": base_path / "labels" / "val",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return dirs


def generate_random_box(img_size: int, min_size: float = 0.1, max_size: float = 0.4) -> tuple:
    """
    Generate a random bounding box within the image.
    
    Returns:
        tuple: (x_center, y_center, width, height) normalized to [0, 1]
    """
    # Random width and height (as fraction of image size)
    w = random.uniform(min_size, max_size)
    h = random.uniform(min_size, max_size)
    
    # Random center position (ensure box stays within image)
    x_center = random.uniform(w / 2, 1 - w / 2)
    y_center = random.uniform(h / 2, 1 - h / 2)
    
    return x_center, y_center, w, h


def normalized_to_pixel(x_center: float, y_center: float, w: float, h: float, img_size: int) -> tuple:
    """Convert normalized YOLO coordinates to pixel coordinates for drawing."""
    x1 = int((x_center - w / 2) * img_size)
    y1 = int((y_center - h / 2) * img_size)
    x2 = int((x_center + w / 2) * img_size)
    y2 = int((y_center + h / 2) * img_size)
    return x1, y1, x2, y2


def generate_image_and_label(
    img_size: int,
    num_classes: int,
    num_objects: int = None,
) -> tuple:
    """
    Generate a synthetic image with colored rectangles and corresponding labels.
    
    Args:
        img_size: Size of the square image
        num_classes: Number of classes
        num_objects: Number of objects to place (random if None)
    
    Returns:
        tuple: (PIL.Image, list of label strings)
    """
    # Create a random background
    bg_color = (
        random.randint(200, 255),
        random.randint(200, 255),
        random.randint(200, 255),
    )
    img = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Determine number of objects
    if num_objects is None:
        num_objects = random.randint(1, 3)
    
    labels = []
    
    for _ in range(num_objects):
        # Random class
        class_id = random.randint(0, num_classes - 1)
        
        # Random box (normalized)
        x_center, y_center, w, h = generate_random_box(img_size)
        
        # Convert to pixel coordinates for drawing
        x1, y1, x2, y2 = normalized_to_pixel(x_center, y_center, w, h, img_size)
        
        # Draw filled rectangle with class color
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
        
        # Create YOLO format label: class_id x_center y_center width height
        label = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        labels.append(label)
    
    return img, labels


def create_dataset(
    base_path: Path,
    num_train: int,
    num_val: int,
    img_size: int,
    num_classes: int,
) -> None:
    """Create the complete dummy dataset."""
    print(f"\nCreating dummy dataset at: {base_path}")
    print(f"  Train images: {num_train}")
    print(f"  Val images: {num_val}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Number of classes: {num_classes}")
    print()
    
    # Create directories
    dirs = create_directories(base_path)
    
    # Generate training data
    print("\nGenerating training data...")
    for i in range(num_train):
        img, labels = generate_image_and_label(img_size, num_classes)
        
        # Save image
        img_path = dirs["train_images"] / f"train_{i:04d}.jpg"
        img.save(img_path, "JPEG", quality=95)
        
        # Save labels
        label_path = dirs["train_labels"] / f"train_{i:04d}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(labels))
        
        if (i + 1) % 10 == 0 or i == num_train - 1:
            print(f"  Generated {i + 1}/{num_train} training samples")
    
    # Generate validation data
    print("\nGenerating validation data...")
    for i in range(num_val):
        img, labels = generate_image_and_label(img_size, num_classes)
        
        # Save image
        img_path = dirs["val_images"] / f"val_{i:04d}.jpg"
        img.save(img_path, "JPEG", quality=95)
        
        # Save labels
        label_path = dirs["val_labels"] / f"val_{i:04d}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(labels))
        
        print(f"  Generated {i + 1}/{num_val} validation samples")
    
    print("\nDataset creation complete!")
    print(f"\nDataset structure:")
    print(f"  {base_path}/")
    print(f"  +-- images/")
    print(f"  |   +-- train/  ({num_train} images)")
    print(f"  |   +-- val/    ({num_val} images)")
    print(f"  +-- labels/")
    print(f"      +-- train/  ({num_train} label files)")
    print(f"      +-- val/    ({num_val} label files)")


def main():
    parser = argparse.ArgumentParser(
        description="Create a dummy dataset for YOLOv5 training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the dataset (default: data/dummy relative to project root)",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=DEFAULT_NUM_TRAIN,
        help=f"Number of training images (default: {DEFAULT_NUM_TRAIN})",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=DEFAULT_NUM_VAL,
        help=f"Number of validation images (default: {DEFAULT_NUM_VAL})",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Image size (square) in pixels (default: {DEFAULT_IMG_SIZE})",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help=f"Number of object classes (default: {DEFAULT_NUM_CLASSES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine output path
    if args.output:
        base_path = Path(args.output)
    else:
        # Default: data/dummy relative to project root
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        base_path = project_root / "data" / "dummy"
    
    # Create the dataset
    create_dataset(
        base_path=base_path,
        num_train=args.num_train,
        num_val=args.num_val,
        img_size=args.img_size,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
