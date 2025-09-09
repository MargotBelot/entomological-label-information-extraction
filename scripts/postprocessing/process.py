# Import third-party libraries
import json
import os
import argparse
import sys
import time
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import the necessary module from the 'label_processing' and `label_postprocessing` module packages
import label_processing.utils as utils
from label_postprocessing.ocr_postprocessing import (
    is_empty,
    is_nuri,
    is_plausible_prediction,
    save_transcripts,
    correct_transcript
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'process.py [-h] -j <ocr json> -o <output directory>'

    parser = argparse.ArgumentParser(
        description="Execute the ocr_postprocessing.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-j', '--json',
            metavar='',
            type=str,
            required = True,
            help=('Path to ocr output json file.')
            )

    parser.add_argument(
            '-o', '--outdir',
            metavar='',
            type=str,
            required = True,
            help=('Output directory where files should be saved.')
            )

    return parser.parse_args()

def process_ocr_output(ocr_output: str, outdir: str) -> None:
    """
    Process OCR output to identify Nuri labels, empty labels, and correct plausible labels.

    Args:
        ocr_output (str): Path to the OCR output JSON file.
        outdir (str): Directory to save processed files.
    """
    start_time = time.time()
    nuri_labels, empty_labels, plausible_labels, clean_labels = {}, {}, [], []
    
    try:
        with open(ocr_output, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {ocr_output} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {ocr_output}.")
        return
    
    for label in labels:
        label_id = label.get("ID", "Unknown")
        text = label.get("text", "")
        
        if is_nuri(text):
            nuri_labels[label_id] = text
        elif is_empty(text):
            empty_labels[label_id] = ""
        elif is_plausible_prediction(text):
            plausible_labels.append({"ID": label_id, "text": text})
            clean_labels.append({"ID": label_id, "text": correct_transcript(text)})
    
    os.makedirs(outdir, exist_ok=True)
    save_transcripts(nuri_labels, os.path.join(outdir, "identifier.csv"))
    save_transcripts(empty_labels, os.path.join(outdir, "empty_transcripts.csv"))
    utils.save_json(plausible_labels, "plausible_transcripts.json", outdir)
    utils.save_json(clean_labels, "corrected_transcripts.json", outdir)
    
    print(f"Finished in {round(time.perf_counter() - start_time, 2)} seconds")

def main():
    """
    Main function to parse arguments and execute OCR processing.
    """
    args = parse_arguments()
    process_ocr_output(args.json, args.outdir)

if __name__ == "__main__":
    main()
