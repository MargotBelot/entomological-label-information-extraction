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
            [-c <c_value>] -d <crop-dir> [-multi <multiprocessing>] -o <outdir> [-o <out-dir>]'
    
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


def ocr_on_file(
    file_path: str,
    new_dir: str,
    blocksize: int,
    c_value: int,
    thresh_mode: Threshmode,
    tesseract: Tesseract,
) -> tuple[dict[str, str], bool, bool]:
    """
    Perform OCR on a single image file.
    Args:
        file_path (str): Path to the image file.
        new_dir (str): Directory to save preprocessed images.
        blocksize (int): Blocksize for image preprocessing.
        c_value (int): C value for image preprocessing.
        thresh_mode (Threshmode): Thresholding mode for image preprocessing.
        tesseract (Tesseract): Tesseract OCR instance.
    Returns:
        tuple: A tuple containing the transcript dictionary, a boolean indicating if a QR code was detected,
               and a boolean indicating if a NURI format was detected.
    """
    try:
        image = ImageProcessor.read_image(file_path)
        qr_detected, nuri_detected = False, False

        if blocksize:
            image.blocksize(blocksize)
        if c_value:
            image.c_value(c_value)

        decoded_qr = image.read_qr_code()
        if decoded_qr:
            transcript = {"ID": image.filename, "text": decoded_qr}
            qr_detected = True
        else:
            image = image.preprocessing(thresh_mode)
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


def ocr_on_dir(
    crop_dir: str,
    new_dir: str,
    thresholding: int,
    blocksize: int,
    c_value: int,
    multiprocessing: bool,
    verbose: bool = False,
) -> list[dict[str, str]]:
    """
    Perform OCR on all images in a directory.

    Args:
        crop_dir (str): Directory containing cropped images.
        new_dir (str): Directory to save preprocessed images.
        thresholding (int): Thresholding mode for image preprocessing.
        blocksize (int): Blocksize for image preprocessing.
        c_value (int): C value for image preprocessing.
        multiprocessing (bool): Whether to use multiprocessing.
        verbose (bool): Whether to print verbose output.
    Returns:
        list[dict[str, str]]: List of dictionaries containing OCR results.
    """
    tesseract = Tesseract()
    ocr_results = []
    count_qr, total_nuri = 0, 0
    thresh_mode = Threshmode(thresholding)
    file_paths = glob.glob(os.path.join(crop_dir, "*.jpg"))

    if not file_paths:
        print("Error: No JPG files found in the specified directory.")
        return []

    if multiprocessing:
        with mp.Pool() as pool:
            results = pool.starmap(ocr_on_file, [(
                file_path,
                new_dir,
                blocksize,
                c_value,
                thresh_mode,
                tesseract,
            ) for file_path in file_paths])
            for transcript, qr, nuri in results:
                ocr_results.append(transcript)
                count_qr += qr
                total_nuri += nuri
    else:
        for file_path in file_paths:
            transcript, qr, nuri = ocr_on_file(
                file_path=file_path,
                new_dir=new_dir,
                blocksize=blocksize,
                c_value=c_value,
                thresh_mode=thresh_mode,
                tesseract=tesseract,
            )
            ocr_results.append(transcript)
            count_qr += qr
            total_nuri += nuri

    if verbose:
        print(f"QR-codes read: {count_qr}")
        print(f"get_nuri: {total_nuri}")

    return ocr_results



def run_ocr_with_tesseract(
    crop_dir: str,
    outdir: str,
    thresholding: int,
    blocksize: int,
    c_value: int,
    multiprocessing: bool,
    verbose: bool = False,
) -> None:
    """
    Main function to parse arguments and execute OCR with Tesseract.
    
    Args:
        crop_dir (str): Directory containing cropped images.
        outdir (str): Directory to save preprocessed images.
        thresholding (int): Thresholding mode for image preprocessing.
        blocksize (int): Blocksize for image preprocessing.
        c_value (int): C value for image preprocessing.
        multiprocessing (bool): Whether to use multiprocessing.
        verbose (bool): Whether to print verbose output.

    Raises:
        NotADirectoryError: If the cropped images directory does not exist.
    """
    start_time = time.time()

    find_tesseract()

    if verbose:
        print("Tesseract successfully detected.")

    if not validate_paths(crop_dir, outdir):
        exit(1)

    new_dir = utils.generate_filename(crop_dir, "preprocessed")
    new_dir_path = os.path.join(outdir, new_dir)
    Path(new_dir_path).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Performing OCR on {os.path.abspath(crop_dir)}.")

    result_data = ocr_on_dir(
        crop_dir=crop_dir,
        new_dir=new_dir_path,
        thresholding=thresholding,
        multiprocessing=multiprocessing,
        blocksize=blocksize,
        c_value=c_value,
        verbose=verbose,
    )

    if result_data:
        if verbose:
            print(f"Saving results in {os.path.abspath(outdir)}.")

        utils.save_json(result_data, FILENAME, outdir)

    print(f"Finished in {round(time.perf_counter() - start_time, 2)} seconds")


def main():
    """
    Main function to parse arguments and execute OCR with Tesseract.

    Raises:
        NotADirectoryError: If the cropped images directory does not exist.
    """
    args = parse_arguments()

    run_ocr_with_tesseract(
        crop_dir=args.dir,
        outdir=args.outdir,
        thresholding=args.thresholding,
        blocksize=args.blocksize,
        c_value=args.c_value,
        multiprocessing=args.multiprocessing,
        verbose=args.verbose,
    )



if __name__ == "__main__":
    main()
