#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os
import glob
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Callable
import warnings
import time

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import the necessary module from the 'label_processing' module package
from label_processing.text_recognition import (Tesseract, 
                                               ImageProcessor,
                                               Threshmode,
                                               find_tesseract,
                                               )
from label_processing import utils

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

# Constant
FILENAME = "ocr_preprocessed.json"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'tesseract.py [-h] [-v] [-t <thresholding>] [-b <blocksize>] \
            [-c <c_value>] [--clahe] [--normalize-illumination] [--clahe-clip-limit <float>] \
            -d <crop-dir> [-multi <multiprocessing>] -o <outdir>'
    
    parser = argparse.ArgumentParser(
        description="Execute the text_recognition.py module.",
        add_help = False,
        usage = usage)
    
    parser.add_argument(
            '-h','--help',
            action='help',
            help='Description of the command-line arguments.'
            )

    parser.add_argument(
            '-v', '--verbose',
            metavar='',
            action=argparse.BooleanOptionalAction,
            type = int,
            default = False,
            help=('Optional argument: Let the script run verbose')
            )

    parser.add_argument(
            '-t', '--thresholding',
            metavar='',
            choices = (1, 2, 3),
            type=int,
            default = 1,
            action='store',
            help=('Optional argument: select which thresholding should be used primarily.\n'
                 '1 : Otsu\'s thresholding.\n'
                 '2 : Adaptive mean thresholding.\n'
                 '3 : Gaussian adaptive thresholding.\n'
                 'Default is otsus.')
            )
    
    parser.add_argument(
            '-b', '--blocksize',
            metavar='',
            action="store",
            type = int,
            default = None,
            help=('Optional argument: blocksize parameter for adaptive thresholding.')
            )
    
    parser.add_argument(
            '-c', '--c_value',
            metavar='',
            action="store",
            type = int,
            default = None,
            help=('Optional argument: c_value parameter for adaptive thresholding.')
            )
    
    parser.add_argument(
            '--clahe',
            action=argparse.BooleanOptionalAction,
            default=False,
            help=('Optional argument: Apply CLAHE for contrast enhancement. '
                  'Useful for low-contrast or faded labels.')
            )
    
    parser.add_argument(
            '--normalize-illumination',
            action=argparse.BooleanOptionalAction,
            default=False,
            help=('Optional argument: Apply illumination normalization. '
                  'Useful for images with shadows or uneven lighting.')
            )
    
    parser.add_argument(
            '--clahe-clip-limit',
            metavar='',
            action="store",
            type=float,
            default=2.0,
            help=('Optional argument: CLAHE clip limit (default: 2.0). '
                  'Higher values give more contrast.')
            )
    
    parser.add_argument(
            '-o', '--outdir',
            metavar='',
            type=str,
            required = True,
            help=('Directory where the json should be saved')
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
            '-multi', '--multiprocessing',
            metavar='',
            action=argparse.BooleanOptionalAction,
            default=False,
            help=('Select whether to use multiprocessing')
            )
    
    return parser.parse_args()

def validate_paths(crop_dir: str, outdir: str) -> bool:
    """
    Validate the existence of directories.
    
    Args:
        crop_dir (str): Path to the cropped images directory.
        outdir (str): Path to the output directory.
    
    Returns:
        bool: True if paths are valid, False otherwise.
    """
    if not os.path.exists(crop_dir):
        print(f"Error: Input directory '{crop_dir}' does not exist.")
        return False
    if not os.path.exists(outdir):
        print(f"Warning: Output directory '{outdir}' does not exist. Creating it now.")
        os.makedirs(outdir)
    return True

def ocr_on_file(file_path: str, args: argparse.Namespace, thresh_mode: Threshmode, tesseract: Tesseract, new_dir: str) -> tuple[dict[str, str], bool, bool]:
    """
    Perform OCR on a single image file.
    Args:
        file_path (str): Path to the image file.
        args (argparse.Namespace): Parsed command-line arguments.
        thresh_mode (Threshmode): Thresholding mode for image preprocessing.
        tesseract (Tesseract): Tesseract OCR instance.
        new_dir (str): Directory to save preprocessed images.
    Returns:
        tuple: A tuple containing the transcript dictionary, a boolean indicating if a QR code was detected,
               and a boolean indicating if a NURI format was detected.
    """
    try:
        image = ImageProcessor.read_image(file_path)
        qr_detected, nuri_detected = False, False

        if args.blocksize:
            image.blocksize(args.blocksize)
        if args.c_value:
            image.c_value(args.c_value)

        decoded_qr = image.read_qr_code()
        if decoded_qr:
            transcript = {"ID": image.filename, "text": decoded_qr}
            qr_detected = True
        else:
            image = image.preprocessing(
                thresh_mode,
                use_clahe=args.clahe,
                normalize_illum=args.normalize_illumination,
                clahe_clip_limit=args.clahe_clip_limit,
            )
            image.save_image(new_dir)
            tesseract.image = image
            transcript = tesseract.image_to_string()

            if utils.check_nuri_format(transcript["text"]):
                nuri_detected = True
                transcript = utils.replace_nuri(transcript)

        return transcript, qr_detected, nuri_detected
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return {"ID": file_path, "text": "ERROR"}, False, False

def ocr_on_dir(crop_dir: str, new_dir: str, verbose_print: Callable, args: argparse.Namespace) -> list[dict[str, str]]:
    """
    Perform OCR on all images in a directory.
    Args:
        crop_dir (str): Directory containing cropped images.
        new_dir (str): Directory to save preprocessed images.
        verbose_print (Callable): Function to print verbose messages.
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        list[dict[str, str]]: List of dictionaries containing OCR results.
    """
    tesseract = Tesseract()
    ocr_results = []
    count_qr, total_nuri = 0, 0
    thresh_mode = Threshmode(args.thresholding)
    files = glob.glob(os.path.join(crop_dir, "*.jpg"))

    if not files:
        print("Error: No JPG files found in the specified directory.")
        return []

    if args.multiprocessing:
        with mp.Pool() as pool:
            results = pool.starmap(ocr_on_file, [(file, args, thresh_mode, tesseract, new_dir) for file in files])
            for transcript, qr, nuri in results:
                ocr_results.append(transcript)
                count_qr += qr
                total_nuri += nuri
    else:
        for file in files:
            transcript, qr, nuri = ocr_on_file(file, args, thresh_mode, tesseract, new_dir)
            ocr_results.append(transcript)
            count_qr += qr
            total_nuri += nuri

    verbose_print(f"QR-codes read: {count_qr}")
    verbose_print(f"get_nuri: {total_nuri}")
    return ocr_results

if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()
    verbose_print = print if args.verbose else lambda *a, **k: None

    find_tesseract()
    verbose_print("Tesseract successfully detected.")

    if not validate_paths(args.dir, args.outdir):
        exit(1)

    new_dir = utils.generate_filename(args.dir, "preprocessed")
    new_dir_path = os.path.join(args.outdir, new_dir)
    Path(new_dir_path).mkdir(parents=True, exist_ok=True)

    verbose_print(f"Performing OCR on {os.path.abspath(args.dir)}.")
    result_data = ocr_on_dir(args.dir, new_dir_path, verbose_print, args)

    if result_data:
        verbose_print(f"Saving results in {os.path.abspath(args.outdir)}.")
        utils.save_json(result_data, FILENAME, args.outdir)

    print(f"Finished in {round(time.perf_counter() - start_time, 2)} seconds")
