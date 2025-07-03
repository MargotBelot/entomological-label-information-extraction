# Import the necessary module from the 'label_evaluation' module package
import label_evaluation.accuracy_classifier

# Import third-party libraries
import argparse
import os
import warnings
import pandas as pd
import time

# Suppress warning messages during execution
warnings.filterwarnings('ignore')


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'classifiers_eval.py [-h] -o <path to outputs> -d <path to gt_dataframe>'

    parser = argparse.ArgumentParser(
        description="Execute the accuracy_classifier.py module.",
        add_help = False,
        usage = usage)

    parser.add_argument('-h','--help',
                        action='help',
                        help='Description of the command-line arguments.')

    parser.add_argument('-o', '--out_dir',
                        metavar='',
                        type=str,
                        default=os.getcwd(),
                        help=('Directory in which the accuracy scores and plots will be stored. '
                              'Default is the current working directory.'))

    parser.add_argument('-d', '--df',
                        metavar='',
                        type=str,
                        required=True,
                        help=('Path to the input ground truth CSV file.'))

    return parser.parse_args()

def main():
    """
    Main function to evaluate classifier accuracy and generate reports.
    """
    start_time = time.time()
    args = parse_arguments()
    out_dir = args.out_dir
    
    try:
        df = pd.read_csv(args.df, sep=';')
        if "pred" not in df.columns or "gt" not in df.columns:
            raise ValueError("CSV file must contain 'pred' and 'gt' columns.")
    except FileNotFoundError:
        print(f"Error: File '{args.df}' not found.")
        return
    except pd.errors.ParserError:
        print(f"Error: Failed to parse CSV file '{args.df}'.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    pred = df["pred"]
    gt = df["gt"]
    target = gt.unique().tolist()
    
    try:
        label_evaluation.accuracy_classifier.metrics(target, pred, gt, out_dir=out_dir)
        label_evaluation.accuracy_classifier.cm(target, pred, gt, out_dir=out_dir)
    except Exception as e:
        print(f"Error during classification evaluation: {e}")
        return
    
    print(f"Finished in {round(time.time() - start_time, 2)} seconds")

if __name__ == '__main__':
    main()
