# Import third-party libraries
import os
import re
import json
import pandas as pd
from typing import Optional
import numpy as np
import cv2

# Constant
PATTERN = r"(/u/|http|coll|mfn|URI)"

#---------------------Check dir JPEG---------------------#

def check_dir(directory: str) -> None:
    """
    Checks if the directory given as an argument contains jpg files.

    Args:
        directory (str): path to directory

    Raises:
        FileNotFoundError: raised if no jpg files are found in the directory
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    
    if not any(file_name.endswith('.jpg') for file_name in os.listdir(directory)):
        raise FileNotFoundError("The directory given does not contain any jpg files. You might have chosen the wrong directory?")

#---------------------New Filename Preprocessed Images---------------------#

def generate_filename(original_path: str, appendix: str, extension: Optional[str] = None) -> str:
    """
    Gets the path to a file or directory as an input and returns it with an appendix added to the end.

    Args:
        original_path (str): original path to file or directory
        appendix (str): what needs to be appended
        extension (Optional[str]): either no extension (for directories) or a file extension as a string

    Returns:
        str: new file or directory name
    """
    # Remove extension if it has one
    new_filename, _ = os.path.splitext(os.path.basename(original_path))
    
    appendix = appendix.strip("_")
    if original_path.endswith(os.path.sep):
        new_filename = f"{os.path.basename(os.path.dirname(new_filename))}_{appendix}"
    else:
        new_filename = f"{new_filename}_{appendix}"
    
    if extension:
        if extension[0] != ".":
            new_filename = f"{new_filename}.{extension}"
        else:
            new_filename = f"{new_filename}{extension}"
    
    return new_filename

#---------------------Save JSON---------------------#

def save_json(data: list[dict], filename: str, path: str) -> None:
    """
    Saves a json file with human-readable format.

    Args:
        data (list[dict]): output of the OCR
        filename (str): name for the json file
        path (str): path where the json should be saved
    """
    filepath = os.path.join(path, filename)
    with open(filepath, "w", encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ': '))

#---------------------Check and Correct NURIs---------------------#

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

#---------------------Load CSV and JPG Files---------------------#

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
    with open(file, 'r') as f:
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