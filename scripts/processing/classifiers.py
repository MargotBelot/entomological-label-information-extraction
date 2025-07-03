#!/usr/bin/env python3

# Import the necessary module from the 'label_processing' module package
import label_processing.tensorflow_classifier

# Import third-party libraries
import argparse
import os
import warnings
import time
import logging
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the classification script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'python classifiers.py -m  <select model> -j <path to jpgs> -o  <path to output directory>'

    parser = argparse.ArgumentParser(
        description="Classify JPEG images using a pre-trained TensorFlow model.",
        usage=usage,
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument(
        '-m', '--model',
        type=int,
        choices=range(1, 4),
        help=(
            'Select a built-in classifier model:\n'
            '  1: identifier / not_identifier\n'
            '  2: handwritten / printed\n'
            '  3: multi-label / single-label'
        )
    )

    parser.add_argument(
        '-j', '--jpg_dir',
        type=str,
        required=True,
        help='Directory containing input JPEG images.'
    )

    parser.add_argument(
        '-o', '--out_dir',
        type=str,
        default=os.getcwd(),
        help='Directory to store outputs (default: current working directory).'
    )

    return parser.parse_args()


def resolve_default_model_path(model_int: int) -> str:
    """
    Get the default model path based on model number.

    Args:
        model_int (int): Model number (1-3)

    Returns:
        str: Absolute path to the default model
    """
    script_dir = os.path.dirname(__file__)
    model_paths = {
        1: "../../models/label_classifier_identifier_not_identifier",
        2: "../../models/label_classifier_hp",
        3: "../../models/label_classifier_multi_single"
    }

    model_rel_path = model_paths.get(model_int)
    if not model_rel_path:
        raise ValueError(f"No default model path found for model {model_int}")
    return os.path.abspath(os.path.join(script_dir, model_rel_path))


def get_class_names_by_model(model_int: int) -> list[str]:
    """
    Return default class names for the selected model number.

    Args:
        model_int (int): Model number (1-3)

    Returns:
        list[str]: Class labels
    """
    class_names = {
        1: ["not_identifier", "identifier"],
        2: ["handwritten", "printed"],
        3: ["multi", "single"]
    }
    return class_names.get(model_int)


def load_class_names_from_file(path: str) -> list[str]:
    """
    Load class names from a text file (one per line).

    Args:
        path (str): Path to the class names file.

    Returns:
        list[str]: List of class names.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Classes file not found at: {path}")
    with open(path, "r") as f:
        return f.read().splitlines()


def main() -> None:
    """
    Main function to execute classification using a TensorFlow model.
    """
    start_time = time.time()
    args = parse_arguments()

    if not os.path.isdir(args.jpg_dir):
        raise NotADirectoryError(f"JPEG input directory does not exist: {args.jpg_dir}")

    if args.model:
        model_path = resolve_default_model_path(args.model)
        class_names = get_class_names_by_model(args.model)
        if not class_names:
            raise ValueError(f"No class names found for model {args.model}")
    elif args.model_path and args.classes_path:
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        class_names = load_class_names_from_file(args.classes_path)
    else:
        raise ValueError("You must provide either a model number (-m) or a model path (-p) with a class names file (-c).")

    logging.info("Loading model...")
    model = label_processing.tensorflow_classifier.get_model(model_path)

    logging.info("Classifying images...")
    df = label_processing.tensorflow_classifier.class_prediction(
        model=model,
        class_names=class_names,
        jpg_dir=args.jpg_dir,
        out_dir=args.out_dir
    )

    logging.info("Saving classified images...")
    label_processing.tensorflow_classifier.filter_pictures(
        jpg_dir=args.jpg_dir,
        dataframe=df,
        out_dir=args.out_dir
    )

    duration = time.time() - start_time
    logging.info(f"Classification completed in {duration:.2f} seconds.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)