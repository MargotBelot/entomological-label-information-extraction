# Import third-party libraries
import json
import nltk
from nltk import word_tokenize
nltk.download('punkt', quiet=True)
import string
import pandas as pd
import re
from typing import List, Dict

# Constants
NON_ASCII = re.compile(" [^\x00-\x7F] ")
NON_ALPHA_NUM = re.compile("[^a-zA-Z\d\s]{2,}")
PIPE = re.compile("[|]")

def count_mean_token_length(tokens: List[str]) -> float:
    """
    Calculates the mean length of tokens in a list.

    Args:
        tokens (list): List of tokens.

    Returns:
        float: Mean token length.
    """
    if not tokens:
        return 0
    total_length = sum(len(token) for token in tokens)
    mean_length = total_length / len(tokens)
    return round(mean_length, 2)

def is_plausible_prediction(transcript: str) -> bool:
    """
    Checks if a transcript is a plausible prediction based on the average token length.

    Args:
        transcript (str): Input transcript.

    Returns:
        bool: True if the transcript is plausible, False otherwise.
    """
    try:
        tokens = word_tokenize(transcript)
        tokens_no_punct = [token for token in tokens if token not in string.punctuation]
        return count_mean_token_length(tokens_no_punct) >= 2
    except Exception as e:
        print(f"Error checking plausible prediction: {e}")
        return False

def correct_transcript(transcript: str) -> str:
    """
    Performs corrections on a transcript, removing non-ASCII characters, multiple non-alphanumeric characters,
    the pipe character, and other special symbols (like °, ', , etc.). Also removes any trailing periods.
    
    Args:
        transcript (str): Input transcript.
    
    Returns:
        str: Corrected transcript.
    """
    try:
        # Remove non-ASCII characters
        transcript = re.sub(NON_ASCII, ' ', transcript)
        
        # Remove non-alphanumeric characters (like special symbols, except for spaces)
        transcript = re.sub(NON_ALPHA_NUM, '', transcript)
        
        # Remove the pipe character
        transcript = re.sub(PIPE, '', transcript)
        
        # Remove specific characters (degree symbol, apostrophes, commas)
        transcript = transcript.replace('°', '').replace("'", '').replace(",", "")
        
        # Remove any trailing periods
        return transcript.rstrip(".")
    except Exception as e:
        print(f"Error correcting transcript: {e}")
        return transcript

def is_nuri(transcript: str) -> bool:
    """
    Checks if a transcript starts with "http," indicating a Nuri.

    Args:
        transcript (str): Input transcript.

    Returns:
        bool: True if the transcript is a Nuri, False otherwise.
    """
    return transcript.startswith("http")

def is_empty(transcript: str) -> bool:
    """
    Checks if a transcript is empty.

    Args:
        transcript (str): Input transcript.

    Returns:
        bool: True if the transcript is empty, False otherwise.
    """
    return len(transcript.strip()) == 0

def save_transcripts(transcripts: Dict, file_name: str) -> None:
    """
    Saves transcripts as a CSV file.

    Args:
        transcripts (dict): Dictionary of transcripts.
        file_name (str): Name of the output CSV file.
    """
    try:
        pd.DataFrame.from_dict(transcripts, orient="index").to_csv(file_name)
    except Exception as e:
        print(f"Error saving transcripts to CSV: {e}")

def save_json(transcripts: List[Dict], file_name: str) -> None:
    """
    Saves transcripts as a JSON file.

    Args:
        transcripts (list): List of transcripts.
        file_name (str): Name of the output JSON file.
    """
    try:
        with open(file_name, "w") as outfile:
            json.dump(transcripts, outfile, indent=4)
    except Exception as e:
        print(f"Error saving JSON file: {e}")

def process_ocr_output(ocr_output: str) -> None:
    """
    Processes OCR output, categorizing and saving transcripts based on Nuri, empty, plausible, and corrected.

    Args:
        ocr_output (str): OCR output file path.
    """
    try:
        nuri_labels, empty_labels, plausible_labels, clean_labels = {}, {}, [], []
        
        with open(ocr_output, 'r') as f:
            labels = json.load(f)
            
        for label in labels:
            text = label.get("text", "")
            label_id = label.get("ID", "")
            
            if is_nuri(text):
                nuri_labels[label_id] = text
            elif is_empty(text):
                empty_labels[label_id] = ""
            elif is_plausible_prediction(text):
                plausible_labels.append({"ID": label_id, "text": text})
                clean_labels.append({"ID": label_id, "text": correct_transcript(text)})
        
        save_transcripts(nuri_labels, "nuris.csv")
        save_transcripts(empty_labels, "empty_transcripts.csv")
        save_json(plausible_labels, "plausible_transcripts.json")
        save_json(clean_labels, "corrected_transcripts.json")
    except Exception as e:
        print(f"Error processing OCR output: {e}")
