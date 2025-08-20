#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import glob
import os
import warnings
import time
import cv2  # OpenCV for QR code detection
from google.cloud import vision
from google.oauth2 import service_account

# Import the necessary module from the 'label_processing' module package
from label_processing import utils

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

# Constants
RESULTS_JSON = "ocr_google_vision.json"
RESULTS_JSON_BOUNDING = "ocr_google_vision_wbounding.json"
BACKUP_TSV = "ocr_google_vision_backup.tsv"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments, including input directories,
        credentials file, output directory, and verbosity flag.
    """
    usage = 'vision.py [-h] [-np] -d <crop dir> -c <credentials> -o <output dir> -v <verbose>'

    parser = argparse.ArgumentParser(
        description="Execute the vision.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Description of the command-line arguments.'
            )
    
    parser.add_argument(
            '-c', '--credentials',
            metavar='',
            type=str,
            required = True,
            help=('Path to the google credentials json file.')
            )
    
    parser.add_argument(
            '-d', '--dir',
            metavar='',
            type=str,
            required = True,
            help=('Directory which contains the cropped jpgs on which the'
                  'ocr is supposed to be applied')
            )
    
    parser.add_argument(
        '-o', '--output_dir',
        metavar='',
        type=str,
        required=True,
        help='Directory where the JSON outputs will be saved.'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output.'
    )

    return parser.parse_args()


def vision_caller(filename: str, client: vision.ImageAnnotatorClient, output_dir: str, verbose: bool) -> dict[str, str]:
    """
    Perform OCR on an image file using Google Cloud Vision API.

    Args:
        filename (str): Path to the image file.
        client (vision.ImageAnnotatorClient): Initialized Vision API client.
        output_dir (str): Directory where the backup TSV file will be saved.
        verbose (bool): Flag to enable verbose output.

    Returns:
        dict[str, str]: A dictionary containing the OCR result with 'ID', 'text', and 'bounding_boxes'.
    """
    if verbose:
        print(f"Processing file: {filename}")
    try:
        with open(filename, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if not texts:
            raise ValueError("No text detected")
        ocr_result = {
            "ID": os.path.basename(filename),
            "text": texts[0].description,
            "bounding_boxes": [[(v.x, v.y) for v in text.bounding_poly.vertices] for text in texts]
        }
    except Exception as e:
        if verbose:
            print(f"OCR failed for {filename}: {e}")
        return {"ID": os.path.basename(filename), "text": "", "bounding_boxes": [], "error": str(e)}
    backup_file = os.path.join(output_dir, BACKUP_TSV)
    with open(backup_file, "a", encoding="utf8") as bf:
        bf.write(f"{ocr_result['ID']}	{ocr_result['text']}")
    return ocr_result

def detect_qr_code(image_path: str, verbose: bool) -> bool:
    """
    Detect if an image contains a QR code.

    Args:
        image_path (str): Path to the image file.
        verbose (bool): Flag to enable verbose output.

    Returns:
        bool: True if a QR code is detected, False otherwise.
    """
    if not os.path.isfile(image_path):
        if verbose:
            print(f"File not found: {image_path}")
        return False
    image = cv2.imread(image_path)
    if image is None:
        if verbose:
            print(f"Error reading image: {image_path}")
        return False
    qr_detector = cv2.QRCodeDetector()
    try:
        data, _, _ = qr_detector.detectAndDecode(image)
        return bool(data)
    except cv2.error as e:
        if verbose:
            print(f"QR detection error in {image_path}: {e}")
        return False

def main(crop_dir: str, credentials: str, output_dir: str, encoding: str = 'utf8', verbose: bool = False) -> None:
    """
    Perform OCR on all JPEG images in a directory using Google Cloud Vision API.

    Args:
        crop_dir (str): Directory containing the JPEG images to process.
        credentials (str): Path to the Google Cloud Vision API credentials JSON file.
        output_dir (str): Directory where the JSON outputs will be saved.
        encoding (str, optional): Encoding to use for saving files. Defaults to 'utf8'.
        verbose (bool, optional): Flag to enable verbose output. Defaults to False.
    """
    start_time = time.time()
    print("Starting OCR process...")
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        client = vision.ImageAnnotatorClient(credentials=service_account.Credentials.from_service_account_file(credentials))
    except Exception as e:
        print(f"Failed to initialize Google Vision API client: {e}")
        return
    utils.check_dir(crop_dir)
    filenames = glob.glob(os.path.join(crop_dir, "*.jpg"))
    if verbose:
        print(f"Total files found: {len(filenames)}")
    filenames = [file for file in filenames if not detect_qr_code(file, verbose)]
    if verbose:
        print(f"Files to process after QR filtering: {len(filenames)}")
    results_json = [vision_caller(filename, client, output_dir, verbose) for filename in filenames]
    print("OCR process completed. Saving results...")
    utils.save_json(results_json, RESULTS_JSON_BOUNDING, output_dir)
    json_no_bounding = [{k: v for k, v in entry.items() if k != "bounding_boxes"} for entry in results_json]
    utils.save_json(json_no_bounding, RESULTS_JSON, output_dir)
    print(f"Finished in {round(time.perf_counter() - start_time, 2)} seconds")

if __name__ == '__main__':
    args = parse_arguments()
    main(args.dir, args.credentials, args.output_dir, verbose=args.verbose)