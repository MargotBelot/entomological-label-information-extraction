#!/usr/bin/env python3

# Import third-party libraries
import argparse
import os
import warnings
import pandas as pd
import plotly.io as pio
import time

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

# Import the necessary module from the 'label_evaluation' module package
from label_evaluation import iou_scores

# Constants
FILENAME_CSV = "iou_scores.csv"
FILENAME_BOXPLOT = "iou_box.png"
FILENAME_BARCHART = "class_pred.png"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'detection_eval.py [-h] -g <ground truth coordinates> -p <predicted coordinates> -r <results>'

    parser = argparse.ArgumentParser(
        description="Execute the iou_scores.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-g', '--ground_truth_coord',
            metavar='',
            type=str,
            required = True,
            help=('Path to the ground truth coordinates csv.')
            )

    parser.add_argument(
            '-p', '--predicted_coord',
            metavar='',
            type=str,
            required = True,
            help=('Path to the predicted coordinates csv.')
            )

    parser.add_argument(
            '-r', '--results',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Target folder where the iou accuracy results and plots are saved.\n'
                  'Default is the user current working directory.')
            )

    
    return parser.parse_args()

def main():
    """
    Main function to evaluate IOU scores and generate visualizations.
    """
    start_time = time.time()
    args = parse_arguments()
    result_dir = args.results
    
    for file_path in [args.ground_truth_coord, args.predicted_coord]:
        if not os.path.isfile(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    try:
        df_gt = pd.read_csv(args.ground_truth_coord)
        df_pred = pd.read_csv(args.predicted_coord)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return
    except Exception as e:
        print(f"Unexpected error reading CSV files: {e}")
        return
    
    # Compute IOU scores and save results
    try:
        df_concat = iou_scores.concat_frames(df_gt, df_pred)
        os.makedirs(result_dir, exist_ok=True)
        csv_filepath = os.path.join(result_dir, FILENAME_CSV)
        df_concat.to_csv(csv_filepath, index=False)
        print(f"IoU scores CSV saved at: {csv_filepath}")
        
        # Create box plot and save
        fig = iou_scores.box_plot_iou(df_concat, accuracy_txt_path=os.path.join(result_dir, 'accuracy_percentage.txt'))
        boxplot_filepath = os.path.join(result_dir, FILENAME_BOXPLOT)
        pio.write_image(fig, boxplot_filepath, format="png", width=1800, height=1200, scale=1)
        print(f"Box plot saved at: {boxplot_filepath}")
        
        print(f"Finished in {round(time.time() - start_time, 2)} seconds")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return

if __name__ == "__main__":
    main()
