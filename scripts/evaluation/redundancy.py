#!/usr/bin/env python3

# Import the necessary module from the 'label_evaluation' module package
import label_evaluation.redundancy

# Import third-party libraries
import argparse
import json
import warnings
import os
import time

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

# Constant
FILENAME_TXT = "percentage_redundancy.txt"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'redundancy.py [-h] -d <dataset dir> -o <output>'

    parser = argparse.ArgumentParser(
        description="Execute the redundancy.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-d', '--dataset_dir',
            metavar='',
            type=str,
            required = True,
            help=('Path to the dataset containing labels transcriptions.')
            )
            
    parser.add_argument(
            '-o', '--output',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Target folder where the text file with the redundancy result is saved\n'
                  'Default is the user current working directory.')
            )

    return parser.parse_args()

def main():
    """
    Main function to evaluate redundancy in a dataset and save results.
    """
    start_time = time.time()
    args = parse_arguments()
    
    if not os.path.isfile(args.dataset_dir):
        print(f"Error: Dataset file not found - {args.dataset_dir}")
        return
    
    try:
        with open(args.dataset_dir, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file {args.dataset_dir}")
        return
    except Exception as e:
        print(f"Unexpected error reading dataset: {e}")
        return
    
    try:
        result = label_evaluation.redundancy.per_redundancy(json_data)
    except Exception as e:
        print(f"Error during redundancy evaluation: {e}")
        return
    
    out_dir = os.path.realpath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        filepath = os.path.join(out_dir, FILENAME_TXT)
        with open(filepath, "w", encoding='utf-8') as text_file:
            text_file.write(f"{result}%")
        print(f"Redundancy result successfully saved at {filepath}")
    except Exception as e:
        print(f"Error writing results to file: {e}")
        return
    
    print(f"Finished in {round(time.time() - start_time, 2)} seconds")

if __name__ == "__main__":
    main()
