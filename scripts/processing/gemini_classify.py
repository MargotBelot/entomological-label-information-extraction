#!/usr/bin/env python3
"""
Gemini Classification + Rotation Script

Replaces the traditional pipeline steps (empty detection, identifier classification,
handwritten/printed classification, rotation correction) with a single Gemini API call
per specimen image.

For each specimen image:
1. Reads detection results (bounding boxes) from the detection CSV
2. Sends the full image + bounding boxes to Gemini
3. Gets per-label: category (empty/identifier/printed/handwritten/mixed) + rotation angle
4. Rotates labels locally using OpenCV
5. Saves classified images into category subdirectories and rotation metadata

Usage:
    python gemini_classify.py -i <input_dir> -o <output_dir> -d <detection_csv>
    python gemini_classify.py -i <input_dir> -o <output_dir>  # SLI mode (no detection)
"""

import argparse
import csv
import json
import os
import sys
import time
import cv2
import pandas as pd
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from label_processing.gemini_processor import (
    get_client,
    classify_and_detect_rotation,
    detect_and_classify,
    rotate_image,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify and rotate labels using Gemini API.",
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True,
        help="Directory containing specimen images.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="Output directory for classified and rotated images.",
    )
    parser.add_argument(
        "-d", "--detection-csv", type=str, default=None,
        help="Path to detection CSV (input_predictions.csv). "
             "If not provided, treats each input image as a single label (SLI mode).",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Gemini API key. Falls back to GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--gemini-detection", action="store_true",
        help="Use Gemini for label detection instead of a detection CSV. "
             "Gemini will find bounding boxes AND classify labels in one call.",
    )
    return parser.parse_args()


def load_detection_results(csv_path: str) -> dict[str, list[dict]]:
    """
    Load detection bounding boxes grouped by source filename.

    Args:
        csv_path: Path to the detection CSV file.

    Returns:
        Dict mapping filename -> list of bounding box dicts.
    """
    df = pd.read_csv(csv_path)
    grouped = {}

    for _, row in df.iterrows():
        filename = row["filename"]
        if filename not in grouped:
            grouped[filename] = []

        grouped[filename].append({
            "label_index": len(grouped[filename]) + 1,
            "xmin": int(float(row["xmin"])),
            "ymin": int(float(row["ymin"])),
            "xmax": int(float(row["xmax"])),
            "ymax": int(float(row["ymax"])),
        })

    return grouped


def process_sli_image(image_path: str, client, model: str) -> list[dict]:
    """
    Process a single pre-cropped label image (SLI mode).

    The entire image is treated as one label.

    Args:
        image_path: Path to the label image.
        client: Gemini client.
        model: Gemini model name.

    Returns:
        List with one classification result dict.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read {image_path}")
        return []

    h, w = img.shape[:2]
    bboxes = [{"label_index": 1, "xmin": 0, "ymin": 0, "xmax": w, "ymax": h}]
    return classify_and_detect_rotation(image_path, bboxes, client, model)


def save_rotated_label(
    image: cv2.typing.MatLike,
    bbox: dict,
    angle: float,
    category: str,
    output_dir: Path,
    source_stem: str,
    label_index: int,
) -> str:
    """
    Crop label region from the full image, rotate it, and save to category dir.

    Args:
        image: Full specimen image.
        bbox: Bounding box dict with xmin, ymin, xmax, ymax.
        angle: Rotation angle in degrees (clockwise).
        category: Label category for subdirectory.
        output_dir: Base output directory.
        source_stem: Source image filename stem.
        label_index: Label index for naming.

    Returns:
        Output filename.
    """
    # Crop the label region from the full image
    xmin = max(0, bbox["xmin"])
    ymin = max(0, bbox["ymin"])
    xmax = min(image.shape[1], bbox["xmax"])
    ymax = min(image.shape[0], bbox["ymax"])
    label_crop = image[ymin:ymax, xmin:xmax]

    # Rotate if needed
    if abs(angle) >= 0.5:
        label_crop = rotate_image(label_crop, angle)

    # Save to category subdirectory
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{source_stem}_{label_index}.jpg"
    out_path = cat_dir / out_filename
    cv2.imwrite(str(out_path), label_crop)

    return out_filename


def main():
    start_time = time.time()
    args = parse_arguments()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Gemini client
    print("Initializing Gemini client...")
    client = get_client(args.api_key)

    # Determine mode
    use_gemini_detection = args.gemini_detection
    is_mli = args.detection_csv is not None or use_gemini_detection
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

    if use_gemini_detection:
        print("MLI mode: using Gemini for detection + classification")
        detection_results = None  # Will call Gemini per image
    elif args.detection_csv is not None:
        print(f"MLI mode: using detection results from {args.detection_csv}")
        detection_results = load_detection_results(args.detection_csv)
    else:
        print("SLI mode: treating each image as a single label")
        detection_results = None

    # Collect images to process
    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions and not f.name.startswith("._")
    )
    print(f"Found {len(image_files)} images to process")

    # Prepare metadata CSV
    metadata_path = output_dir / "gemini_classification.csv"
    all_results = []

    for i, image_path in enumerate(image_files, 1):
        filename = image_path.name
        stem = image_path.stem
        print(f"[{i}/{len(image_files)}] Processing {filename}...")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Warning: Could not read {filename}, skipping")
            continue

        # Get labels: either Gemini detection, CSV detection, or SLI
        if use_gemini_detection:
            # Gemini does detection + classification in one call
            try:
                labels = detect_and_classify(
                    str(image_path), client, args.model
                )
            except Exception as e:
                print(f"  Error calling Gemini detection for {filename}: {e}")
                continue
            if not labels:
                print(f"  No labels detected by Gemini for {filename}, skipping")
                continue
        else:
            # Get bounding boxes for this image
            if is_mli:
                bboxes = detection_results.get(filename, [])
                if not bboxes:
                    print(f"  No detections for {filename}, skipping")
                    continue
            else:
                # SLI: whole image is one label
                h, w = image.shape[:2]
                bboxes = [{"label_index": 1, "xmin": 0, "ymin": 0, "xmax": w, "ymax": h}]

            # Call Gemini for classification + rotation
            try:
                labels = classify_and_detect_rotation(
                    str(image_path), bboxes, client, args.model
                )
            except Exception as e:
                print(f"  Error calling Gemini for {filename}: {e}")
                continue

        # Process each classified label
        h, w = image.shape[:2]
        for label in labels:
            category = label.get("category", "unknown")
            angle = label.get("rotation_angle", 0)
            confidence = label.get("confidence", 0)
            label_idx = label.get("label_index", 1)

            # Bounding box: detect_and_classify() already returns pixel coords;
            # for CSV-detection / SLI the coords are also in pixels.
            bbox = {
                "xmin": label.get("xmin", 0),
                "ymin": label.get("ymin", 0),
                "xmax": label.get("xmax", w),
                "ymax": label.get("ymax", h),
            }

            # Save rotated label to category directory
            out_filename = save_rotated_label(
                image, bbox, angle, category, output_dir, stem, label_idx
            )

            # Record metadata
            all_results.append({
                "source_image": filename,
                "label_filename": out_filename,
                "label_index": label_idx,
                "category": category,
                "rotation_angle": angle,
                "confidence": confidence,
                "xmin": bbox["xmin"],
                "ymin": bbox["ymin"],
                "xmax": bbox["xmax"],
                "ymax": bbox["ymax"],
            })

            print(f"  Label {label_idx}: {category} (angle={angle}°, conf={confidence:.2f})")

    # Save classification metadata
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(metadata_path, index=False)
        print(f"\nClassification metadata saved to {metadata_path}")

        # Print summary
        print("\n=== Classification Summary ===")
        for cat in ["empty", "identifier", "printed", "handwritten", "mixed"]:
            count = len(df[df["category"] == cat])
            if count > 0:
                print(f"  {cat}: {count}")
        print(f"  Total labels: {len(df)}")

    # Also save as JSON for consolidation
    json_path = output_dir / "gemini_classification.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    duration = time.time() - start_time
    print(f"\nFinished in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
