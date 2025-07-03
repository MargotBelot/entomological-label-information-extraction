#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os
import time
import warnings
from pathlib import Path

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

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
    usage = 'detection.py [-h] [-c N] [-np N] -o <path to jpgs outputs> -j <path to jpgs>'

    parser = argparse.ArgumentParser(
        description="Execute the label_detection_module.py.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Description of the command-line arguments.'
            )
    
    parser.add_argument(
            '-o', '--out_dir',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Directory in which the resulting crops and the csv will be stored.\n'
                  'Default is the user current working directory.')
            )
    
    parser.add_argument(
            '-j', '--jpg_dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the jpgs are stored.')
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

    script_dir = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(script_dir, "../../models/label_detection_model.pth")
    jpg_dir = Path(args.jpg_dir)
    out_dir = args.out_dir
    classes = ["label"]

    if not validate_paths(jpg_dir, out_dir, MODEL_PATH):
        return

    try:
        predictor = scrop.PredictLabel(MODEL_PATH, classes)
        df = scrop.prediction_parallel(jpg_dir, predictor, PROCESSES)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    if df.empty:
        print("No valid predictions were generated. Skipping further processing.")
        return

    try:
        df = scrop.clean_predictions(jpg_dir, df, THRESHOLD, out_dir=out_dir)
    except Exception as e:
        print(f"Error cleaning predictions: {e}")
        return

    print(f"Processing finished in {round(time.perf_counter() - start_time, 2)} seconds")

    try:
        create_crops(jpg_dir, df, out_dir=out_dir)
    except Exception as e:
        print(f"Error during cropping: {e}")
        return

    print(f"Finished in {round(time.perf_counter() - start_time, 2)} seconds")


if __name__ == '__main__':
    main()
