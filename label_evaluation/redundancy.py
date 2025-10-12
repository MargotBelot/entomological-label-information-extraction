# Import third-party library
import warnings

# Suppress warning messages during execution
warnings.filterwarnings("ignore")


def clean_data(data: list[dict]) -> list[dict]:
    """
    Preprocess the dataset by converting text to lowercase, removing punctuation and whitespace,
    and excluding entries containing 'http'.

    Args:
        data (list of dict): List of dictionaries with labels' transcription.

    Returns:
        list of dict: Preprocessed list of dictionaries.
    """
    try:
        cleaned_data = []
        for item in data:
            if "text" not in item:
                continue
            text = item["text"].lower()
            cleaned_text = "".join(
                e for e in text if e.isalnum() or e.isspace()
            ).replace(" ", "")
            if "http" not in cleaned_text:
                item["text"] = cleaned_text
                cleaned_data.append(item)
        return cleaned_data
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return []


def redundancy(data: list[dict]) -> list[dict]:
    """
    Identify duplicate entries in a preprocessed dataset.

    Args:
        data (list of dict): Preprocessed list of dictionaries with labels' transcription.

    Returns:
        list of dict: List of dictionaries containing duplicate entries.
    """
    try:
        data = clean_data(data)
        text_set = set()
        duplicates = []
        for item in data:
            text = item["text"]
            if text in text_set:
                duplicates.append(item)
            text_set.add(text)
        return duplicates
    except Exception as e:
        print(f"Error identifying redundant entries: {e}")
        return []


def per_redundancy(data: list[dict]) -> int:
    """
    Calculate the percentage of transcription redundancy in a dataset.

    Args:
        data (list of dict): Preprocessed list of dictionaries with labels' transcription.

    Returns:
        int: Percentage of redundant text.
    """
    try:
        data_clean = clean_data(data)
        duplicates = redundancy(data_clean)
        sum_text = len(data_clean)
        sum_dup = len(duplicates)
        return round((sum_dup / sum_text) * 100) if sum_text > 0 else 0
    except Exception as e:
        print(f"Error calculating redundancy percentage: {e}")
        return 0
