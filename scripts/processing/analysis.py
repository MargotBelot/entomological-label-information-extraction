# Import third-party libraries
import os
import argparse
import sys
import time

# Import the necessary module from the 'label_processing' module package
from label_processing.detect_empty_labels import find_empty_labels


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'analysis.py [-h] -o <output image dir> -i <input image dir>'

    parser = argparse.ArgumentParser(
        description="Execute the detect_empty_labels_module.py.",
        add_help=False,
        usage=usage
    )

    parser.add_argument(
        '-h', '--help',
        action='help',
        help='Description of the command-line arguments.'
    )

    parser.add_argument(
        '-o', '--output_image_dir',
        metavar='',
        type=str,
        default=os.getcwd(),
        help=('Directory where the filtered images will be stored.\n'
              'Default is the user current working directory.')
    )

    parser.add_argument(
        '-i', '--input_image_dir',
        metavar='',
        type=str,
        required=True,
        help=('Directory where the input jpgs are stored.')
    )

    return parser.parse_args()

def validate_directories(input_dir: str, output_dir: str) -> None:
    """
    Validate that the specified directories exist.
    If either directory does not exist, print an error message and exit the program.
    
    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
    """
    for directory, name in [(input_dir, "Input"), (output_dir, "Output")]:
        if not os.path.exists(directory):
            print(f"Error: {name} directory '{directory}' not found.")
            sys.exit(1)

def main():
    """
    Main execution function.
    Parses command-line arguments, validates directories, processes images, and prints the execution duration.
    """
    start_time = time.time()
    args = parse_arguments()
    
    validate_directories(args.input_image_dir, args.output_image_dir)
    
    find_empty_labels(args.input_image_dir, args.output_image_dir)
    print(f"\nEmpty and non-empty labels moved to respective folders in {args.output_image_dir}")
    
    print(f"Finished in {round(time.perf_counter() - start_time, 2)} seconds")

if __name__ == "__main__":
    main()
