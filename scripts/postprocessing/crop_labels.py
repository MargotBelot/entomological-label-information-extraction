#!/usr/bin/env python3
"""
Crop Labels Script

Crops label regions from original specimen images using bounding-box data
from consolidated_results.json or entity_master.json.

Usage:
    python crop_labels.py -i <input_dir> -o <output_dir>
    python crop_labels.py -i <input_dir> -o <output_dir> --source entity_master.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop label regions from original images using JSON bounding boxes.",
    )
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True,
        help="Directory containing original specimen images.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="Pipeline output directory (contains JSON sources and will receive crops).",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="JSON file to read bounding boxes from. "
             "Defaults to entity_master.json if it exists, else consolidated_results.json.",
    )
    parser.add_argument(
        "--crop-dir", type=str, default="cropped_labels",
        help="Subdirectory name inside output-dir for cropped images (default: cropped_labels).",
    )
    return parser.parse_args()


def _labels_from_entity_master(data: list) -> list:
    """Flatten entity_master.json (grouped by source image) into a label list."""
    labels = []
    for entry in data:
        for lbl in entry.get("labels", []):
            if "source_image" not in lbl:
                lbl["source_image"] = entry.get("source_image", "")
            labels.append(lbl)
    return labels


def crop_labels(input_dir: str, output_dir: str, source_path: str, crop_dir_name: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Resolve JSON source
    if source_path:
        src = Path(source_path)
        if not src.is_absolute():
            src = output_path / src
    else:
        # Prefer entity_master.json, fall back to consolidated_results.json
        src = output_path / "entity_master.json"
        if not src.exists():
            src = output_path / "consolidated_results.json"

    if not src.exists():
        print(f"  No JSON source found at {src} — nothing to crop.")
        return

    print(f"Reading bounding boxes from {src.name}")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalise into a flat list of labels
    if isinstance(data, list) and data and "labels" in data[0]:
        labels = _labels_from_entity_master(data)
    elif isinstance(data, list):
        labels = data
    else:
        print("Error: Unrecognised JSON structure.")
        sys.exit(1)

    crop_path = output_path / crop_dir_name
    crop_path.mkdir(parents=True, exist_ok=True)

    crop_count = 0
    skipped = 0
    cache: dict[str, any] = {}  # cache loaded images

    for lbl in labels:
        img_name = lbl.get("source_image", "")
        if not img_name:
            skipped += 1
            continue

        bbox = lbl.get("bbox", {})
        if not bbox:
            skipped += 1
            continue

        # Load image (cached)
        if img_name not in cache:
            img_file = input_path / img_name
            if not img_file.exists():
                cache[img_name] = None
            else:
                cache[img_name] = cv2.imread(str(img_file))
        img = cache[img_name]
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        x1 = max(0, int(bbox.get("xmin", 0)))
        y1 = max(0, int(bbox.get("ymin", 0)))
        x2 = min(w, int(bbox.get("xmax", 0)))
        y2 = min(h, int(bbox.get("ymax", 0)))

        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue

        crop = img[y1:y2, x1:x2]

        stem = Path(img_name).stem
        idx = lbl.get("label_index", crop_count + 1)
        cat = lbl.get("category", "")
        suffix = f"_{cat}" if cat else ""
        out_name = f"{stem}_label{idx}{suffix}.jpg"
        cv2.imwrite(str(crop_path / out_name), crop)
        crop_count += 1

    print(f"  Cropped {crop_count} labels → {crop_path}")
    if skipped:
        print(f"  Skipped {skipped} (missing image or bbox)")


def main():
    args = parse_arguments()
    crop_labels(args.input_dir, args.output_dir, args.source, args.crop_dir)


if __name__ == "__main__":
    main()
