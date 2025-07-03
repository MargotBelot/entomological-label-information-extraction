#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os
import time

# Suppress warning messages during execution
import warnings
warnings.filterwarnings('ignore')

# Import the necessary module from the 'label_evaluation' module package
import label_evaluation.evaluate_text


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'ocr_eval.py [-h] -g <ground truth> -p <predicted ocr output> -r <results>'

    parser = argparse.ArgumentParser(
        description="Execute the evaluate_text.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h', '--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-g', '--ground_truth',
            metavar='',
            type=str,
            required = True,
            help=('Path to the ground truth dataset.')
            )

    parser.add_argument(
            '-p', '--predicted_ocr',
            metavar='',
            type=str,
            required = True,
            help=('Path json file OCR output.')
            )

    parser.add_argument(
            '-r', '--results',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Target folder where the accuracy results are saved.\n'
                  'Default is the user current working directory.')
            )

    
    return parser.parse_args()

def main():
    """
    Main function to evaluate OCR predictions and save results.
    """
    start_time = time.time()
    args = parse_arguments()
    
    for file_path in [args.ground_truth, args.predicted_ocr]:
        if not os.path.isfile(file_path):
            print(f"Error: File not found - {file_path}")
            return
    
    out_dir = os.path.realpath(args.results)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        label_evaluation.evaluate_text.evaluate_text_predictions(args.ground_truth, args.predicted_ocr, out_dir)
        print(f"OCR accuracy results successfully saved in {out_dir}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    print(f"Finished in {round(time.time() - start_time, 2)} seconds")

if __name__ == "__main__":
    main()
