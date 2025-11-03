# Import third-party libraries
import os
import re
import json
import pandas as pd
from typing import Optional
import numpy as np
import cv2
import hashlib
from PIL import Image

# Constant
PATTERN = r"(/u/|http|coll|mfn|URI)"

# ---------------------Check dir JPEG---------------------#


def validate_image_integrity(
    filepath: str, max_size_mb: int = 25, max_dimensions: tuple = (8000, 8000)
) -> bool:
    """
    Validate image file integrity with strict memory safety limits.

    Args:
        filepath (str): path to image file
        max_size_mb (int): maximum file size in MB (default: 25MB)
        max_dimensions (tuple): maximum width/height in pixels (default: 8000x8000)

    Returns:
        bool: True if image is valid and safe to process, False otherwise
    """
    try:
        # SECURITY: Strict file size limit to prevent memory exhaustion
        file_size = os.path.getsize(filepath)
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            print(
                f"SECURITY WARNING: Image {filepath} too large ({file_size // (1024*1024)}MB > {max_size_mb}MB)"
            )
            return False

        # Verify image can be opened and check dimensions
        with Image.open(filepath) as img:
            width, height = img.size

            # SECURITY: Check image dimensions to prevent memory bombs
            if width > max_dimensions[0] or height > max_dimensions[1]:
                print(
                    f"SECURITY WARNING: Image {filepath} dimensions too large ({width}x{height} > {max_dimensions[0]}x{max_dimensions[1]})"
                )
                return False

            # Calculate estimated memory usage (width * height * channels * bytes_per_pixel)
            estimated_memory_mb = (width * height * 3 * 4) / (
                1024 * 1024
            )  # Assume 4 bytes per pixel worst case
            if estimated_memory_mb > 500:  # 500MB memory limit per image
                print(
                    f"SECURITY WARNING: Image {filepath} would use too much memory (~{estimated_memory_mb:.1f}MB)"
                )
                return False

            img.verify()  # This will raise an exception if image is corrupted

        # Additional OpenCV validation (with size check)
        test_img = cv2.imread(filepath)
        if test_img is None:
            return False

        # Double-check OpenCV loaded image dimensions
        cv_height, cv_width = test_img.shape[:2]
        # Allow exact match OR swapped dimensions (EXIF rotation)
        dimensions_ok = (
            (cv_width == width and cv_height == height) or
            (cv_width == height and cv_height == width)
        )
        if not dimensions_ok:
            print(f"SECURITY WARNING: PIL and OpenCV dimension mismatch for {filepath}")
            print(f"  PIL: {width}x{height}, OpenCV: {cv_width}x{cv_height}")
            return False

        return True
    except (IOError, SyntaxError, Exception):
        return False


def check_dir(directory: str) -> None:
    """
    Checks if the directory contains valid jpg files with integrity validation.

    Args:
        directory (str): path to directory

    Raises:
        FileNotFoundError: raised if no valid jpg files are found in the directory
        ValueError: raised if corrupted image files are detected
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    jpg_files = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg"))]

    if not jpg_files:
        raise FileNotFoundError(
            "The directory given does not contain any jpg or jpeg files. You might have chosen the wrong directory?"
        )

    # Validate image integrity
    invalid_files = []
    for jpg_file in jpg_files:
        filepath = os.path.join(directory, jpg_file)
        if not validate_image_integrity(filepath):
            invalid_files.append(jpg_file)

    if invalid_files:
        raise ValueError(
            f"Corrupted or invalid image files detected: {invalid_files[:5]}{'...' if len(invalid_files) > 5 else ''}. Please check and replace these files."
        )


# ---------------------New Filename Preprocessed Images---------------------#


def generate_filename(
    original_path: str, appendix: str, extension: Optional[str] = None
) -> str:
    """
    Gets the path to a file or directory as an input and returns it with an appendix added to the end.

    Args:
        original_path (str): original path to file or directory
        appendix (str): what needs to be appended
        extension (Optional[str]): either no extension (for directories) or a file extension as a string

    Returns:
        str: new file or directory name
    """
    # Convert Path object to string if necessary
    original_path_str = str(original_path)

    # Remove extension if it has one
    new_filename, _ = os.path.splitext(os.path.basename(original_path_str))

    appendix = appendix.strip("_")
    if original_path_str.endswith(os.path.sep):
        new_filename = f"{os.path.basename(os.path.dirname(new_filename))}_{appendix}"
    else:
        new_filename = f"{new_filename}_{appendix}"

    if extension:
        if extension[0] != ".":
            new_filename = f"{new_filename}.{extension}"
        else:
            new_filename = f"{new_filename}{extension}"

    return new_filename


# ---------------------Save JSON---------------------#


def save_json(data: list[dict], filename: str, path: str) -> None:
    """
    Saves a json file with human-readable format.

    Args:
        data (list[dict]): output of the OCR
        filename (str): name for the json file
        path (str): path where the json should be saved
    """
    filepath = os.path.join(path, filename)
    with open(filepath, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(",", ": "))


# ---------------------Check and Correct NURIs---------------------#


def check_nuri_format(transcript: str) -> bool:
    """
    Check NURI's format in OCR transcription "text".

    Args:
        transcript (str): text field from OCR output

    Returns:
        bool: True if NURI pattern found, False otherwise
    """
    # Search for NURI patterns in "text"
    pattern = re.compile(PATTERN)
    match = pattern.search(transcript)
    return bool(match)


def replace_nuri(transcript: dict[str, str]) -> dict[str, str]:
    """
    Correct NURI format in OCR transcription JSON output.

    Args:
        transcript (dict[str, str]): JSON transcript with "ID" and "text" fields.

    Returns:
        dict[str, str]: JSON transcript with corrected NURI formats in "text" field.
    """
    reg_nuri = re.compile(r"_u_[A-Za-z0-9]+")
    reg_picturae_nuri = re.compile(r"_u_([0-9a-fA-F]+)\.jpg")

    try:
        if "ID" in transcript and "text" in transcript:
            nuri = reg_nuri.search(transcript["ID"])
            picturae_nuri = reg_picturae_nuri.search(transcript["ID"])

            if nuri:
                # Replace using the first pattern
                replace_string = f"http://coll.mfn-berlin.de/u/{nuri.group()[3:]}"
                transcript["text"] = replace_string
            elif picturae_nuri:
                # Replace using the second pattern
                replace_string = f"http://coll.mfn-berlin.de/u/{picturae_nuri.group(1)}"
                transcript["text"] = replace_string
    except AttributeError:
        pass

    return transcript


# ---------------------Load CSV and JPG Files---------------------#


def load_dataframe(filepath_csv: str) -> pd.DataFrame:
    """
    Loads the CSV file using Pandas.

    Args:
        filepath_csv (str): path to the CSV file

    Returns:
        pd.DataFrame: The CSV as a Pandas DataFrame
    """
    dataframe = pd.read_csv(filepath_csv)
    return dataframe


def load_jpg(filepath: str) -> np.ndarray:
    """
    Loads the jpg files using the OpenCV module.

    Args:
        filepath (str): path to jpg files

    Returns:
        np.ndarray: OpenCV image object
    """
    jpg = cv2.imread(filepath)
    return jpg


def load_json(file: str) -> dict:
    """
    Load JSON data from a file and deserialize it.

    Args:
        file (str): The name of the file containing JSON data.

    Returns:
        dict: The JSON data as a dictionary
    """
    with open(file, "r") as f:
        data = json.load(f)
    return data


def read_vocabulary(file: str) -> dict:
    """
    Read a CSV file containing vocabulary and convert it to a dictionary.

    Args:
        file (str): The name of the CSV file containing vocabulary data.

    Returns:
        dict: A dictionary where keys and values are taken from the CSV data.
    """
    voc = pd.read_csv(file)
    return dict(voc.values)


# ---------------------Model Integrity Verification---------------------#


def verify_model_integrity(
    model_path: str, checksums_file: str = None, require_checksum: bool = True
) -> bool:
    """
    SECURITY: Mandatory model file integrity verification using SHA256 checksums.

    Args:
        model_path (str): path to model file
        checksums_file (str): path to checksums file (auto-detected if None)
        require_checksum (bool): if True, requires checksum file to exist (default: True)

    Returns:
        bool: True if model integrity is verified, False otherwise

    Raises:
        SecurityError: If model integrity cannot be verified and require_checksum=True
    """
    try:
        if not os.path.exists(model_path):
            raise Exception(f"SECURITY ERROR: Model file not found: {model_path}")

        # Auto-detect checksums file if not provided
        if checksums_file is None:
            model_dir = os.path.dirname(model_path)
            checksums_file = os.path.join(model_dir, "checksums.sha256")

        # SECURITY: Require checksums file for verification
        if require_checksum and not os.path.exists(checksums_file):
            raise Exception(
                f"SECURITY ERROR: Checksums file required but not found: {checksums_file}"
            )

        # Calculate current file hash
        print(f"SECURITY: Calculating SHA256 hash for {model_path}...")
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        current_hash = sha256_hash.hexdigest()

        # Verify against checksums file
        if os.path.exists(checksums_file):
            with open(checksums_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Parse line: "hash  filename"
                    parts = line.split()
                    if len(parts) >= 2:
                        expected_hash = parts[0]
                        filename_in_checksum = " ".join(parts[1:])

                        # Check if this line matches our model file
                        if (
                            model_path in filename_in_checksum
                            or os.path.basename(model_path) in filename_in_checksum
                        ):
                            if current_hash == expected_hash:
                                print(
                                    f"SECURITY: Model integrity VERIFIED for {model_path}"
                                )
                                return True
                            else:
                                print(f"SECURITY ERROR: Hash mismatch for {model_path}")
                                print(f"Expected: {expected_hash}")
                                print(f"Got:      {current_hash}")
                                return False

            # If we get here, model wasn't found in checksums file
            if require_checksum:
                raise Exception(
                    f"SECURITY ERROR: Model {model_path} not found in checksums file {checksums_file}"
                )

        # If no checksums file and require_checksum=False, do basic validation
        if not require_checksum:
            print(
                f"WARNING: No checksums verification for {model_path} - basic validation only"
            )
            return len(current_hash) == 64 and os.path.getsize(model_path) > 1024

        return False

    except Exception as e:
        print(f"SECURITY ERROR: Model integrity verification failed: {e}")
        if require_checksum:
            raise e
        return False
