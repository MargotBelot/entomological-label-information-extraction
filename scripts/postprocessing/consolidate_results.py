#!/usr/bin/env python3
"""
Consolidate Pipeline Results Script

Creates a single JSON file that links all per-image results across the pipeline
(detection → classification → rotation → OCR → post‑processing).

Each entry includes (when available): ``filename``, detection ``coordinates`` and
``confidence``, boolean classification flags (``empty``, ``identifier``, ``handwritten``,
``printed``), rotation metadata (``angle``, ``corrected``), OCR summary (``method``,
``raw_text``, ``confidence``), and post‑processing fields (``cleaned_text``, ``plausible``).
"""

import json
import csv
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    usage = 'consolidate_results.py [-h] -o <output directory> [-f <output filename>]'
    
    parser = argparse.ArgumentParser(
        description="Consolidate all pipeline results into linked JSON output.",
        add_help=False,
        usage=usage
    )
    
    parser.add_argument(
        '-h', '--help',
        action='help',
        help='Open this help text.'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        metavar='',
        type=str,
        required=True,
        help='Output directory containing pipeline results.'
    )
    
    parser.add_argument(
        '-f', '--filename',
        metavar='',
        type=str,
        default='consolidated_results.json',
        help='Output filename for consolidated results (default: consolidated_results.json).'
    )
    
    return parser.parse_args()

def load_detection_results(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load detection results from input_predictions.csv."""
    detection_results = {}
    predictions_file = os.path.join(output_dir, 'input_predictions.csv')
    
    if not os.path.exists(predictions_file):
        print(f"Warning: {predictions_file} not found. Skipping detection results.")
        return detection_results
    
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '')
                if filename:
                    # Parse bounding box coordinates
                    bbox = []
                    confidence = 0.0
                    
                    try:
                        # Handle different CSV formats
                        if 'bbox_coordinates' in row:
                            bbox_str = row.get('bbox_coordinates', '[]')
                            bbox = json.loads(bbox_str.replace("'", '"'))
                        elif all(coord in row for coord in ['xmin', 'ymin', 'xmax', 'ymax']):
                            bbox = [
                                float(row['xmin']), float(row['ymin']),
                                float(row['xmax']), float(row['ymax'])
                            ]
                        
                        if 'confidence_scores' in row:
                            conf_str = row.get('confidence_scores', '[0.0]')
                            conf_list = json.loads(conf_str.replace("'", '"'))
                            confidence = float(conf_list[0]) if conf_list else 0.0
                        elif 'score' in row:
                            confidence = float(row['score'])
                            
                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass
                    
                    detection_results[filename] = {
                        'coordinates': bbox,
                        'confidence': confidence
                    }
    except Exception as e:
        print(f"Error loading detection results: {e}")
    
    return detection_results

def determine_classification_path(filename: str, output_dir: str) -> Dict[str, bool]:
    """Determine which classification path a file took."""
    classification = {
        'empty': False,
        'identifier': False,
        'handwritten': False,
        'printed': False
    }
    
    # Check different classification directories
    directories = {
        'empty': 'empty',
        'identifier': 'identifier', 
        'handwritten': 'handwritten',
        'printed': 'printed'
    }
    
    for category, directory in directories.items():
        dir_path = os.path.join(output_dir, directory)
        if os.path.exists(dir_path):
            # Check if file exists in this directory
            file_pattern = os.path.join(dir_path, filename)
            if os.path.exists(file_pattern) or glob.glob(file_pattern + '*'):
                classification[category] = True
                break
    
    return classification

def load_rotation_results(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load rotation correction results."""
    rotation_results = {}
    
    # First try to load from rotation metadata file (preferred method)
    # Check both main directory and subdirectories (like printed_preprocessed)
    meta_files = [
        os.path.join(output_dir, 'rotation_metadata.csv'),
        os.path.join(output_dir, 'printed_preprocessed', 'rotation_metadata.csv'),
        os.path.join(output_dir, 'printed_rotated', 'rotation_metadata.csv')
    ]
    
    for meta_file in meta_files:
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row.get('filename', '')
                        if filename:
                            angle = int(row.get('angle', 0))
                            corrected = str(row.get('corrected', 'False')).lower() == 'true'
                            rotation_results[filename] = {
                                'angle': angle,
                                'corrected': corrected
                            }
                return rotation_results
            except Exception as e:
                print(f"Error loading rotation metadata from {meta_file}: {e}")
                # Fall back to directory-based detection
    
    # Fallback: directory-based detection (legacy method)
    rotated_dir = os.path.join(output_dir, 'rotated')
    printed_dir = os.path.join(output_dir, 'printed')
    
    if os.path.exists(rotated_dir):
        # Files in rotated directory had rotation correction applied
        for filename in os.listdir(rotated_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                rotation_results[filename] = {
                    'rotation_applied': True,  # Legacy field
                    'original_angle': 'unknown'  # Legacy field
                }
    elif os.path.exists(printed_dir):
        # Files went directly to printed (multi-label pipeline, no rotation step)
        for filename in os.listdir(printed_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                rotation_results[filename] = {
                    'rotation_applied': False,  # Legacy field
                    'original_angle': 'not_applicable'  # Legacy field
                }
    
    return rotation_results

def load_ocr_results(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load OCR results from JSON files."""
    ocr_results = {}
    
    # Check for OCR result files
    ocr_files = [
        'ocr_preprocessed.json',
        'ocr_google_vision.json',
        'ocr_results.json'
    ]
    
    for ocr_file in ocr_files:
        file_path = os.path.join(output_dir, ocr_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                method = 'tesseract' if 'preprocessed' in ocr_file else 'google_vision'
                
                # Handle different OCR result formats
                if isinstance(ocr_data, list):
                    for item in ocr_data:
                        file_id = item.get('ID', '')
                        if file_id:
                            entry = {
                                'method': method,
                                'raw_text': item.get('text', '')
                            }
                            # Include confidence if available
                            if 'confidence' in item:
                                entry['confidence'] = item['confidence']
                            ocr_results[file_id] = entry
                elif isinstance(ocr_data, dict):
                    for file_id, content in ocr_data.items():
                        if isinstance(content, dict):
                            text = content.get('text', '')
                            entry = {
                                'method': method,
                                'raw_text': text
                            }
                            # Include confidence if available
                            if 'confidence' in content:
                                entry['confidence'] = content['confidence']
                        else:
                            text = str(content)
                            entry = {
                                'method': method,
                                'raw_text': text
                            }
                        
                        ocr_results[file_id] = entry
            except Exception as e:
                print(f"Error loading OCR results from {ocr_file}: {e}")
                continue
            break  # Use first available OCR file
    
    return ocr_results

def load_postprocessing_results(output_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load post-processing results."""
    postprocessing_results = {}
    
    # Load different post-processing outputs
    files_to_check = {
        'plausible_transcripts.json': 'plausible',
        'corrected_transcripts.json': 'corrected', 
        'identifier.csv': 'identifier',
        'empty_transcripts.csv': 'empty'
    }
    
    for filename, category in files_to_check.items():
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            try:
                if filename.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            file_id = item.get('ID', '')
                            if file_id:
                                postprocessing_results[file_id] = {
                                    'original_text': item.get('original_text', ''),
                                    'cleaned_text': item.get('text', ''),
                                    'corrected': category == 'corrected',
                                    'plausible': category == 'plausible',
                                    'category': category
                                }
                
                elif filename.endswith('.csv'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        next(reader, None)  # Skip header if present
                        for row in reader:
                            if len(row) >= 2:
                                file_id = row[0]
                                text = row[1] if len(row) > 1 else ''
                                postprocessing_results[file_id] = {
                                    'original_text': text,
                                    'cleaned_text': text,
                                    'corrected': False,
                                    'plausible': False,
                                    'category': category
                                }
            except Exception as e:
                print(f"Error loading post-processing results from {filename}: {e}")
    
    return postprocessing_results

def consolidate_results(output_dir: str) -> List[Dict[str, Any]]:
    """Consolidate all pipeline results for files that completed OCR processing."""
    print("Loading pipeline results...")
    
    # Load all result components
    detection_results = load_detection_results(output_dir)
    rotation_results = load_rotation_results(output_dir)
    ocr_results = load_ocr_results(output_dir)
    postprocessing_results = load_postprocessing_results(output_dir)
    
    print(f"Found {len(detection_results)} detection results")
    print(f"Found {len(rotation_results)} rotation results") 
    print(f"Found {len(ocr_results)} OCR results")
    print(f"Found {len(postprocessing_results)} post-processing results")
    
    consolidated = []
    
    # Only process files that have OCR results (completed the full pipeline)
    for filename in sorted(ocr_results.keys()):
        # Build consolidated result for this file
        result = {
            'filename': filename,
            'detection': detection_results.get(filename, {}),
            'classification': determine_classification_path(filename, output_dir),
            'rotation': rotation_results.get(filename, {}),
            'ocr': ocr_results.get(filename, {}),
            'postprocessing': postprocessing_results.get(filename, {})
        }
        
        # Only include files that have OCR results (ensuring they went through the full pipeline)
        if result['ocr']:
            consolidated.append(result)
    
    return consolidated

def main():
    """Main function to consolidate pipeline results."""
    args = parse_arguments()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist.")
        return
    
    print(f"Consolidating results from: {args.output_dir}")
    
    # Consolidate all results
    consolidated_results = consolidate_results(args.output_dir)
    
    print(f"Consolidated {len(consolidated_results)} files with processing results")
    
    # Save consolidated results
    output_file = os.path.join(args.output_dir, args.filename)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_results, f, indent=2, ensure_ascii=False)
        
        print(f" Consolidated results saved to: {output_file}")
        
        # Print summary
        print("\n=== Processing Summary ===")
        empty_count = sum(1 for r in consolidated_results if r['classification'].get('empty', False))
        identifier_count = sum(1 for r in consolidated_results if r['classification'].get('identifier', False))
        handwritten_count = sum(1 for r in consolidated_results if r['classification'].get('handwritten', False))
        printed_count = sum(1 for r in consolidated_results if r['classification'].get('printed', False))
        ocr_count = sum(1 for r in consolidated_results if r['ocr'])
        
        print(f"Total files processed: {len(consolidated_results)}")
        print(f"Empty labels: {empty_count}")
        print(f"Identifier labels: {identifier_count}")
        print(f"Handwritten labels: {handwritten_count}")
        print(f"Printed labels: {printed_count}")
        print(f"OCR processed: {ocr_count}")
        
    except Exception as e:
        print(f"Error saving consolidated results: {e}")

if __name__ == "__main__":
    main()
