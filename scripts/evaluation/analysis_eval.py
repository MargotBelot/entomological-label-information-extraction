# Import third-party libraries
import os
import argparse
import sys
import time

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'analysis_eval.py [-h] -e <empty folder> -n <not empty folder>'

    parser = argparse.ArgumentParser(
        description="Script for evaluating the empty label detection script.",
        add_help=False,
        usage=usage
    )

    parser.add_argument(
        '-h', '--help',
        action='help',
        help='Description of the command-line arguments.'
    )

    parser.add_argument(
        '-e', '--empty_folder',
        metavar='',
        type=str,
        required=True,
        help=('Directory where the predicted empty labels images are stored.')
    )

    parser.add_argument(
        '-n', '--not_empty_folder',
        metavar='',
        type=str,
        required=True,
        help=('Directory where the predicted not_empty labels images are stored.')
    )

    return parser.parse_args()

def evaluate_labels(empty_folder: str, not_empty_folder: str) -> None:
    """
    Evaluate predicted labels against ground truth labels.

    Args:
        empty_folder (str): Path to directory containing predicted empty labels images.
        not_empty_folder (str): Path to directory containing predicted not empty labels images.
    """
    correct_empty, total_empty = 0, 0
    correct_not_empty, total_not_empty = 0, 0
    
    try:
        for filename in os.listdir(empty_folder):
            total_empty += 1
            label = filename.split("__")[-1].split(".")[0]
            if label == "empty":
                correct_empty += 1
    except FileNotFoundError:
        print(f"Error: Directory '{empty_folder}' not found.")
        sys.exit(1)
    
    try:
        for filename in os.listdir(not_empty_folder):
            total_not_empty += 1
            label = filename.split("__")[-1].split(".")[0]
            if label != "empty":
                correct_not_empty += 1
    except FileNotFoundError:
        print(f"Error: Directory '{not_empty_folder}' not found.")
        sys.exit(1)
    
    empty_accuracy = correct_empty / total_empty if total_empty else 0
    not_empty_accuracy = correct_not_empty / total_not_empty if total_not_empty else 0
    total_correct = correct_empty + correct_not_empty
    total_files = total_empty + total_not_empty
    total_accuracy = total_correct / total_files if total_files else 0
    
    print(f"Empty folder accuracy: {empty_accuracy:.2%} ({correct_empty}/{total_empty})")
    print(f"Not empty folder accuracy: {not_empty_accuracy:.2%} ({correct_not_empty}/{total_not_empty})")
    print(f"Total accuracy: {total_accuracy:.2%} ({total_correct}/{total_files})")

def main():
    """
    Main function to execute label evaluation.
    """
    start_time = time.time()
    args = parse_arguments()
    
    if not os.path.isdir(args.empty_folder):
        print(f"Error: Input directory '{args.empty_folder}' not found.")
        sys.exit(1)
    if not os.path.isdir(args.not_empty_folder):
        print(f"Error: Input directory '{args.not_empty_folder}' not found.")
        sys.exit(1)
    
    evaluate_labels(args.empty_folder, args.not_empty_folder)
    print(f"Finished in {round(time.perf_counter() - start_time, 2)} seconds")

if __name__ == "__main__":
    main()