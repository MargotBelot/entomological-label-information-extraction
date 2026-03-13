#!/usr/bin/env python3
"""
Consolidate Pipeline Results Script

Creates a single JSON file that links all per-label results across the pipeline
(detection → classification → rotation → OCR → post‑processing).

Supports both the traditional (TensorFlow-based) pipeline and the Gemini pipeline.
Output is a flat list of per-label entries, each containing: ``source_image``,
``label_filename``, ``label_index``, ``category``, bounding-box coordinates,
``rotation_angle``, and ``ocr`` (method, text, confidence).
"""

import json
import csv
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import glob

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Consolidate all pipeline results into a single JSON file."
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, required=True,
        help='Output directory containing pipeline results.'
    )
    parser.add_argument(
        '-f', '--filename', type=str, default='consolidated_results.json',
        help='Output filename (default: consolidated_results.json).'
    )
    return parser.parse_args()


# ---- Gemini pipeline loaders ------------------------------------------------

def _load_gemini_classification(output_dir: str) -> List[Dict[str, Any]]:
    """Load per-label classification from gemini_classification.json.

    Returns a list of label dicts with keys: source_image, label_filename,
    label_index, category, rotation_angle, confidence, xmin/ymin/xmax/ymax.
    """
    json_path = os.path.join(output_dir, 'gemini_classification.json')
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"Warning: could not load {json_path}: {e}")
    return []


def _load_gemini_ocr(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load Gemini OCR results.  Returns {label_filename: {text, confidence}}."""
    path = os.path.join(output_dir, 'ocr_gemini.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = {}
        if isinstance(data, list):
            for item in data:
                fid = item.get('ID', '')
                if fid:
                    results[fid] = {
                        'method': 'gemini',
                        'text': item.get('text', ''),
                        'confidence': item.get('confidence', None)
                    }
        return results
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
        return {}


def _consolidate_gemini(output_dir: str) -> List[Dict[str, Any]]:
    """Build consolidated entries from Gemini pipeline outputs."""
    labels = _load_gemini_classification(output_dir)
    ocr_map = _load_gemini_ocr(output_dir)
    # Also try corrected transcripts (post-processing output)
    corrected = _load_corrected_transcripts(output_dir)

    consolidated = []
    for lbl in labels:
        label_fn = lbl.get('label_filename', '')
        ocr_entry = ocr_map.get(label_fn, {})
        corrected_entry = corrected.get(label_fn, {})

        entry = {
            'source_image': lbl.get('source_image', ''),
            'label_filename': label_fn,
            'label_index': lbl.get('label_index', ''),
            'category': lbl.get('category', 'unknown'),
            'bbox': {
                'xmin': lbl.get('xmin', 0),
                'ymin': lbl.get('ymin', 0),
                'xmax': lbl.get('xmax', 0),
                'ymax': lbl.get('ymax', 0),
            },
            'rotation_angle': lbl.get('rotation_angle', 0),
            'detection_confidence': lbl.get('confidence', None),
            'ocr': {
                'method': ocr_entry.get('method', 'gemini'),
                'text': corrected_entry.get('text', ocr_entry.get('text', '')),
                'raw_text': ocr_entry.get('text', ''),
                'confidence': ocr_entry.get('confidence', None),
            },
        }
        consolidated.append(entry)
    return consolidated


# ---- Traditional pipeline loaders -------------------------------------------

def _load_detection_results(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load detection results from input_predictions.csv."""
    results: Dict[str, Dict[str, Any]] = {}
    path = os.path.join(output_dir, 'input_predictions.csv')
    if not os.path.exists(path):
        return results
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '')
                if not filename:
                    continue
                bbox = []
                confidence = 0.0
                try:
                    if all(k in row for k in ('xmin', 'ymin', 'xmax', 'ymax')):
                        bbox = [float(row['xmin']), float(row['ymin']),
                                float(row['xmax']), float(row['ymax'])]
                    if 'score' in row:
                        confidence = float(row['score'])
                except (ValueError, KeyError):
                    pass
                results.setdefault(filename, []).append({
                    'coordinates': bbox,
                    'confidence': confidence,
                })
    except Exception as e:
        print(f"Warning: could not load {path}: {e}")
    return results


def _determine_classification(filename: str, output_dir: str) -> str:
    """Return the category string for a label file (directory-based check)."""
    for category in ('empty', 'identifier', 'handwritten', 'printed'):
        dir_path = os.path.join(output_dir, category)
        if os.path.exists(dir_path):
            if os.path.exists(os.path.join(dir_path, filename)) or \
               glob.glob(os.path.join(dir_path, filename) + '*'):
                return category
    return 'unknown'


def _load_rotation_results(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load rotation correction metadata."""
    results: Dict[str, Dict[str, Any]] = {}
    meta_files = [
        os.path.join(output_dir, 'rotation_metadata.csv'),
        os.path.join(output_dir, 'printed_preprocessed', 'rotation_metadata.csv'),
        os.path.join(output_dir, 'printed_rotated', 'rotation_metadata.csv'),
    ]
    for meta_file in meta_files:
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        fn = row.get('filename', '')
                        if fn:
                            results[fn] = {
                                'angle': float(row.get('angle', 0)),
                                'corrected': str(row.get('corrected', 'False')).lower() == 'true',
                            }
                return results
            except Exception as e:
                print(f"Warning: could not load {meta_file}: {e}")
    return results


def _load_traditional_ocr(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load OCR results from tesseract / google-vision JSON files."""
    ocr_results: Dict[str, Dict[str, Any]] = {}
    for ocr_file, method in [('ocr_preprocessed.json', 'tesseract'),
                              ('ocr_google_vision.json', 'google_vision'),
                              ('ocr_results.json', 'unknown')]:
        path = os.path.join(output_dir, ocr_file)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    fid = item.get('ID', '')
                    if fid:
                        ocr_results[fid] = {
                            'method': method,
                            'text': item.get('text', ''),
                            'confidence': item.get('confidence', None),
                        }
            break  # use first available file
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return ocr_results


def _load_corrected_transcripts(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load corrected / plausible transcripts from post-processing."""
    results: Dict[str, Dict[str, Any]] = {}
    for name in ('corrected_transcripts.json', 'plausible_transcripts.json'):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        fid = item.get('ID', '')
                        if fid and fid not in results:
                            results[fid] = {'text': item.get('text', '')}
            except Exception:
                pass
    return results


def _consolidate_traditional(output_dir: str) -> List[Dict[str, Any]]:
    """Build consolidated entries from traditional pipeline outputs."""
    detection = _load_detection_results(output_dir)
    rotation = _load_rotation_results(output_dir)
    ocr_map = _load_traditional_ocr(output_dir)
    corrected = _load_corrected_transcripts(output_dir)

    consolidated = []
    # Iterate over labels that have OCR results
    for label_fn in sorted(ocr_map.keys()):
        ocr_entry = ocr_map[label_fn]
        corrected_entry = corrected.get(label_fn, {})
        rot = rotation.get(label_fn, {})
        category = _determine_classification(label_fn, output_dir)

        # Try to find the source image from detection results
        source_image = ''
        det_bbox = {}
        det_conf = None
        for src_img, det_list in detection.items():
            # Detection CSV groups labels under the source image filename
            if label_fn.startswith(Path(src_img).stem):
                source_image = src_img
                break

        entry = {
            'source_image': source_image,
            'label_filename': label_fn,
            'label_index': '',
            'category': category,
            'bbox': {},
            'rotation_angle': rot.get('angle', 0),
            'detection_confidence': det_conf,
            'ocr': {
                'method': ocr_entry.get('method', ''),
                'text': corrected_entry.get('text', ocr_entry.get('text', '')),
                'raw_text': ocr_entry.get('text', ''),
                'confidence': ocr_entry.get('confidence', None),
            },
        }
        consolidated.append(entry)
    return consolidated


# ---- Main consolidation ------------------------------------------------------

def consolidate_results(output_dir: str) -> List[Dict[str, Any]]:
    """Auto-detect pipeline type and consolidate all results."""
    print("Loading pipeline results...")

    # Prefer Gemini outputs when available
    gemini_json = os.path.join(output_dir, 'gemini_classification.json')
    if os.path.exists(gemini_json):
        print("Detected Gemini pipeline outputs")
        results = _consolidate_gemini(output_dir)
    else:
        print("Detected traditional pipeline outputs")
        results = _consolidate_traditional(output_dir)

    print(f"Consolidated {len(results)} label entries")
    return results


def main():
    """Main entry point."""
    args = parse_arguments()

    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist.")
        return

    print(f"Consolidating results from: {args.output_dir}")
    consolidated = consolidate_results(args.output_dir)

    output_file = os.path.join(args.output_dir, args.filename)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {output_file}")

        # Summary
        categories = {}
        for r in consolidated:
            cat = r.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        ocr_count = sum(1 for r in consolidated if r.get('ocr', {}).get('text'))

        print(f"\n=== Summary ===")
        print(f"Total labels: {len(consolidated)}")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        print(f"Labels with OCR text: {ocr_count}")
    except Exception as e:
        print(f"Error saving consolidated results: {e}")


if __name__ == "__main__":
    main()
