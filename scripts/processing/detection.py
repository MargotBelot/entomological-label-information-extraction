#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import pandas as pd

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

# Import project configuration
from label_processing.config import get_model_path, get_project_root

# Import the necessary module from the 'label_processing' module package
import label_processing.label_detection as scrop
from label_processing.label_detection import create_crops
from detecto.core import Model

# Constants
THRESHOLD = 0.8
PROCESSES = 12

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Execute label detection on entomological specimen images."
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-j', '--input-dir',
        type=str,
        help='Directory containing specimen images'
    )
    input_group.add_argument(
        '-i', '--input-image',
        type=str,
        help='Single image file to process'
    )
    
    # Output directory (required)
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directory where results will be saved'
    )
    
    # Optional parameters
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Number of images processed simultaneously (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for processing (default: auto)'
    )

    return parser.parse_args()

def validate_paths(jpg_dir: Path, out_dir: str, model_path: str) -> bool:
    """
    Validate the existence of directories and model file.
    
    Args:
        jpg_dir (Path): Path to the input directory.
        out_dir (str): Path to the output directory.
        model_path (str): Path to the model file.
    
    Returns:
        bool: True if all paths are valid, False otherwise.
    """
    if not jpg_dir.exists():
        print(f"Error: The input directory '{jpg_dir}' does not exist.")
        return False
    if not os.path.exists(out_dir):
        print(f"Warning: The output directory '{out_dir}' does not exist. Creating it now.")
        os.makedirs(out_dir)
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return False
    return True

def main():
    """
    Main execution function. Loads the model, processes images,
    filters predictions, and creates image crops.
    """
    start_time = time.perf_counter()
    args = parse_arguments()

    # Use centralized configuration for model path
    try:
        MODEL_PATH = get_model_path("detection")
    except Exception as e:
        print(f"Error getting model path: {e}")
        print("Please ensure the model file exists or set the ENTOMOLOGICAL_DETECTION_MODEL_PATH environment variable.")
        return
    
    # Handle input (directory or single file)
    if args.input_dir:
        jpg_dir = Path(args.input_dir)
        input_type = "directory"
    else:
        # Single file input
        single_file = Path(args.input_image)
        if not single_file.exists():
            print(f"Error: Input file '{single_file}' does not exist.")
            return
        jpg_dir = single_file.parent
        input_type = "single_file"
        print(f"Processing single file: {single_file.name}")
    
    out_dir = args.output_dir
    confidence_threshold = args.confidence
    batch_size = args.batch_size
    device = args.device
    classes = ["label"]

    # Validate paths
    if not os.path.exists(out_dir):
        print(f"Creating output directory: {out_dir}")
        os.makedirs(out_dir)
    if not MODEL_PATH.exists():
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print(f"Expected path: {MODEL_PATH}")
        print("Please ensure the model file exists or set the ENTOMOLOGICAL_DETECTION_MODEL_PATH environment variable.")
        return
    if input_type == "directory" and not jpg_dir.exists():
        print(f"Error: Input directory '{jpg_dir}' does not exist.")
        return

    print(f"Using confidence threshold: {confidence_threshold}")
    print(f"Using batch size: {batch_size}")
    print(f"Using device: {device}")

    try:
        # Initialize predictor (device selection happens in model loading)
        predictor = scrop.PredictLabel(MODEL_PATH, classes)
        
        if input_type == "single_file":
            # Process single file by creating temporary directory structure
            print(f"Processing single file: {single_file}")
            df = predictor.class_prediction(single_file)
            if df.empty:
                # Create empty dataframe with expected columns if no predictions
                df = pd.DataFrame(columns=['filename', 'class', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])
        else:
            # Process directory with parallel processing
            # Adjust number of processes based on batch_size if needed
            processes = min(PROCESSES, batch_size) if batch_size < PROCESSES else PROCESSES
            df = scrop.prediction_parallel(jpg_dir, predictor, processes)
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    if df.empty:
        print("No valid predictions were generated. Skipping further processing.")
        return

    try:
        df = scrop.clean_predictions(jpg_dir, df, confidence_threshold, out_dir=out_dir)
    except Exception as e:
        print(f"Error cleaning predictions: {e}")
        return

    print(f"Detection finished in {round(time.perf_counter() - start_time, 2)} seconds")

    try:
        create_crops(jpg_dir, df, out_dir=out_dir)
    except Exception as e:
        print(f"Error during cropping: {e}")
        return

    print(f"\nProcessing completed in {round(time.perf_counter() - start_time, 2)} seconds")
    print(f"Results saved to: {out_dir}")
    print(f"- CSV file: {out_dir}/input_predictions.csv")
    print(f"- Cropped images: {out_dir}/input_cropped/")


if __name__ == '__main__':
    main()
