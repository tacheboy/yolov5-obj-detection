"""
 Preprocess and validate a fire/smoke dataset for YOLOv5 training.
 
 Expected input structure:
   <dataset_root>/
     train/
       images/
       labels/
     val/
       images/
       labels/
     test/            (optional)
       images/
       labels/
 
 Output structure (default):
   data/fire_data/
     images/train, images/val, images/test
     labels/train, labels/val, labels/test
 
 Usage:
   python data/fire_data/preprocess_fire_dataset.py --dataset-root "D:\\Fire Dataset"
   python data/fire_data/preprocess_fire_dataset.py --dataset-root "/kaggle/working/D Fire Dataset"
"""
 
import argparse
import shutil
from pathlib import Path
 
 
SUPPORTED_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}
 
 
def resolve_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    return project_root / output_path

 
def gather_images(images_dir: Path) -> list[Path]:
    return [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]

 
def label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(images_dir)
    return (labels_dir / relative).with_suffix(".txt")
 
 
def validate_label_file(label_path: Path, num_classes: int) -> tuple[list[int], list[tuple[int, str, str]]]:
    counts = [0] * num_classes
    invalid_lines = []
 
    if not label_path.exists():
        return counts, invalid_lines

    for idx, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            invalid_lines.append((idx, raw, "expected 5 values"))
            continue
        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            invalid_lines.append((idx, raw, "non-numeric values"))
            continue
        if class_id < 0 or class_id >= num_classes:
            invalid_lines.append((idx, raw, "class id out of range"))
            continue
        if any(v < 0.0 or v > 1.0 for v in coords):
            invalid_lines.append((idx, raw, "coords outside [0,1]"))
            continue
        counts[class_id] += 1

    return counts, invalid_lines
 
 
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
 
 
def copy_file(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)
 
 
def process_split(
    split: str,
    dataset_root: Path,
    output_root: Path,
    num_classes: int,
    create_empty_labels: bool,
    strict: bool,
) -> dict:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"

    if not images_dir.exists():
        return {
            "split": split,
            "skipped": True,
            "reason": f"missing images dir: {images_dir}",
        }

    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = gather_images(images_dir)
    class_counts = [0] * num_classes
    invalid_label_count = 0
    empty_label_count = 0
    missing_label_count = 0

    for image_path in images:
        label_path = label_path_for_image(image_path, images_dir, labels_dir)

        dst_image = output_images / image_path.relative_to(images_dir)
        copy_file(image_path, dst_image)

        if labels_dir.exists():
            dst_label = output_labels / label_path.relative_to(labels_dir)
        else:
            dst_label = (output_labels / image_path.relative_to(images_dir)).with_suffix(".txt")
        if label_path.exists():
            if dst_label is not None:
                copy_file(label_path, dst_label)
            counts, invalid_lines = validate_label_file(label_path, num_classes)
            for i, v in enumerate(counts):
                class_counts[i] += v
            if invalid_lines:
                invalid_label_count += len(invalid_lines)
                if strict:
                    details = "\n".join(
                        f"{label_path} L{ln}: {reason} -> {raw}"
                        for ln, raw, reason in invalid_lines
                    )
                    raise ValueError(f"Invalid labels in {label_path}:\n{details}")
        else:
            missing_label_count += 1
            if create_empty_labels and dst_label is not None:
                ensure_parent(dst_label)
                dst_label.write_text("", encoding="utf-8")
                empty_label_count += 1

    orphan_labels = []
    if labels_dir.exists():
        for label_path in labels_dir.rglob("*.txt"):
            image_stub = (images_dir / label_path.relative_to(labels_dir)).with_suffix("")
            has_image = any(
                (image_stub.with_suffix(ext)).exists()
                for ext in SUPPORTED_IMAGE_EXTS
            )
            if not has_image:
                orphan_labels.append(label_path)

    return {
        "split": split,
        "skipped": False,
        "images": len(images),
        "missing_labels": missing_label_count,
        "empty_labels_created": empty_label_count,
        "invalid_label_lines": invalid_label_count,
        "class_counts": class_counts,
        "orphan_labels": orphan_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess fire/smoke dataset for YOLOv5")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset root (contains train/val/test folders)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/fire_data",
        help="Output directory for YOLOv5-ready data (default: data/fire_data)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (default: 2)",
    )
    parser.add_argument(
        "--create-empty-labels",
        action="store_true",
        help="Create empty label files for images without labels",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if invalid label lines are found",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = resolve_output_dir(args.output_dir)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    splits = ["train", "val", "test"]
    results = []
    for split in splits:
        results.append(
            process_split(
                split=split,
                dataset_root=dataset_root,
                output_root=output_root,
                num_classes=args.num_classes,
                create_empty_labels=args.create_empty_labels,
                strict=args.strict,
            )
        )

    print("\nPreprocessing summary")
    print(f"  Dataset root: {dataset_root}")
    print(f"  Output dir:   {output_root}")
    print(f"  Num classes:  {args.num_classes}")

    total_counts = [0] * args.num_classes
    for r in results:
        if r.get("skipped"):
            print(f"  - {r['split']}: skipped ({r['reason']})")
            continue
        print(
            f"  - {r['split']}: images={r['images']}, "
            f"missing_labels={r['missing_labels']}, "
            f"empty_labels_created={r['empty_labels_created']}, "
            f"invalid_label_lines={r['invalid_label_lines']}"
        )
        for i, v in enumerate(r["class_counts"]):
            total_counts[i] += v
        if r["orphan_labels"]:
            print(f"    orphan label files: {len(r['orphan_labels'])}")

    print("  Class counts:")
    for class_id, count in enumerate(total_counts):
        print(f"    {class_id}: {count}")


if __name__ == "__main__":
    main()
