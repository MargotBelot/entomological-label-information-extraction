# Import third-party libraries
import jiwer
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from editdistance import eval

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

class EmptyReferenceError(Exception):
    """
    Custom exception for handling cases where the reference string is empty.
    """
    def __init__(self, message=None):
        self.message = message or "The reference string is empty."
        super().__init__(self.message)

def calculate_cer(reference: list, hypothesis: list) -> float:
    """
    Calculate the Character Error Rate (CER) between reference and hypothesis.

    Args:
        reference (list): List of reference (ground truth) strings.
        hypothesis (list): List of hypothesis (predicted) strings.

    Returns:
        float: The computed CER value.
    """
    if not reference or len(reference[0]) == 0:
        return 0.0
    edit_distance = eval(reference[0], hypothesis[0])
    reference_length = len(reference[0])
    return edit_distance / reference_length

def get_gold_transcriptions(filename: str, sep: str = ',') -> dict:
    """
    Load ground truth transcriptions from a CSV file into a dictionary.

    Args:
        filename (str): Path to the CSV file.
        sep (str, optional): Delimiter used in the CSV file. Defaults to ','.

    Returns:
        dict: Dictionary with keys as unique identifiers and values as transcription text.
    """
    gold_transcriptions = {}
    try:
        with open(filename, encoding='utf-8-sig') as file_in:
            csv_reader = csv.reader(file_in, delimiter=sep)
            next(csv_reader)  # Skip header
            for line_number, line in enumerate(csv_reader, start=2):
                if len(line) != 2:
                    print(f"Skipping malformed line {line_number}: {line}")
                    continue
                line = [field.strip() for field in line]
                gold_transcriptions[line[0]] = line[1]
        return gold_transcriptions
    except Exception as e:
        print(f"Error loading ground truth CSV: {e}")
        return {}

def load_json_predictions(filename: str) -> list:
    """
    Load predictions from a JSON file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        list: List of predictions from the JSON file.
    """
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON predictions: {e}")
        return []

def calculate_scores(gold_text: str, predicted_text: str) -> tuple:
    """
    Calculate Word Error Rate (WER) and Character Error Rate (CER) between ground truth and prediction.

    Args:
        gold_text (str): Ground truth transcription.
        predicted_text (str): Predicted transcription.

    Returns:
        tuple: (WER, CER) both rounded to two decimal places.
    """
    gold_text, predicted_text = gold_text.lower(), predicted_text.lower()
    if not gold_text or gold_text.isspace():
        raise EmptyReferenceError()
    all_scores = jiwer.compute_measures(gold_text, predicted_text)
    wer = round(all_scores['wer'], 2)
    cer = round(calculate_cer([gold_text], [predicted_text]), 2)
    return wer, cer

def create_plot(data: list, score_name: str, file_name: str) -> None:
    """
    Create and save a violin plot for the given error scores.

    Args:
        data (list): List of numerical scores to visualize.
        score_name (str): Name of the score (e.g., "CER" or "WER").
        file_name (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame(data, columns=[score_name])
    sns.violinplot(data=df, inner="box", cut=1, palette="Set2")
    plt.axhline(df[score_name].mean(), color='r', linestyle='--', label=f'Mean: {df[score_name].mean():.2f}')
    plt.axhline(df[score_name].median(), color='g', linestyle='-', label=f'Median: {df[score_name].median():.2f}')
    plt.title(f"Distribution of {score_name}", fontsize=16)
    plt.xlabel(score_name, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend()
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"Plot saved as {file_name}")

def evaluate_text_predictions(ground_truth_file: str, predictions_file: str, out_dir: str) -> tuple:
    """
    Evaluate OCR predictions against a ground truth dataset.

    Args:
        ground_truth_file (str): Path to the ground truth CSV file.
        predictions_file (str): Path to the predictions JSON file.
        out_dir (str): Output directory for results.

    Returns:
        tuple: (List of WER scores, List of CER scores)
    """
    try:
        ground_truth = get_gold_transcriptions(ground_truth_file)
        generated_transcriptions = load_json_predictions(predictions_file)
        wers, cers = [], []
        output_csv = f"{out_dir}/ocr_evaluation.csv"
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Gold", "Predicted", "WER", "CER"])
            
            for entry in generated_transcriptions:
                transcript_id = entry["ID"].strip().lower()
                if transcript_id in ground_truth:
                    gold, predicted = ground_truth[transcript_id], entry["text"].strip()
                    try:
                        wer, cer = calculate_scores(gold, predicted)
                        wers.append(wer)
                        cers.append(cer)
                        writer.writerow([entry["ID"], gold, predicted, wer, cer])
                    except EmptyReferenceError as e:
                        print(f"Skipping ID '{entry['ID']}' due to empty reference: {e}")
        
        create_plot(cers, "CERs", f"{out_dir}/cers.png")
        create_plot(wers, "WERs", f"{out_dir}/wers.png")
        return wers, cers
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return [], []
