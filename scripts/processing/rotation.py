# Import third-party libraries
import os
import argparse
import time
import warnings
import sys
import cv2
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import the necessary module from the 'label_processing' module package
from label_processing.label_rotation import predict_angles

# Suppress warning messages during execution
warnings.filterwarnings('ignore')

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'rotation.py [-h] -o <output image dir> -i <input image dir>'

    parser = argparse.ArgumentParser(
        description="Execute label_rotation_module.py.",
        add_help=False,
        usage=usage
    )

    parser.add_argument('-h', '--help', action='help', help='Display help message and exit.')
    parser.add_argument('-o', '--output_image_dir', metavar='', type=str, default=os.getcwd(),
                        help='Directory where rotated images will be stored. Default is current working directory.')
    parser.add_argument('-i', '--input_image_dir', metavar='', type=str, required=True,
                        help='Directory where input images are stored.')

    return parser.parse_args()

def remove_dot_underscore_files(directory: str) -> None:
    """
    Remove all files starting with "._" from the specified directory.
    
    These files are typically macOS-generated resource fork files that can 
    interfere with image processing.

    Args:
        directory (str): The directory where files should be checked and removed.
    """
    for filename in os.listdir(directory):
        if filename.startswith("._"):
            file_path = os.path.join(directory, filename)
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Warning: Could not remove {file_path} - {e}")

def validate_paths(input_dir: str, output_dir: str) -> list:
    """
    Validate input and output directories and retrieve valid images.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where output images will be saved.

    Returns:
        list: List of valid image file paths.
    """
    if not os.path.exists(input_dir) or not os.access(input_dir, os.R_OK):
        print(f"Error: Cannot access input directory '{input_dir}'. Check existence and permissions.")
        return []

    # List valid image files
    image_extensions = ('.jpg', '.jpeg', '.tiff', '.tif', '.png')
    images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if f.lower().endswith(image_extensions) and not f.startswith("._")]

    if not images:
        print(f"Error: No valid image files found in '{input_dir}'.")
        return []

    # Ensure output directory exists and is writable
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Output directory '{output_dir}' did not exist. Created it.")
        except PermissionError:
            print(f"Error: No permission to create output directory '{output_dir}'.")
            return []

    if not os.access(output_dir, os.W_OK):
        print(f"Error: Output directory '{output_dir}' is not writable. Check permissions.")
        return []

    return images

def validate_model_path(model_path: str) -> bool:
    """
    Validate the existence and accessibility of the model file.

    Args:
        model_path (str): Path to the trained model.

    Returns:
        bool: True if the model file exists and is readable, False otherwise.
    """
    if not os.path.exists(model_path) or not os.access(model_path, os.R_OK):
        print(f"Error: Model file '{model_path}' is missing or not readable.")
        return False
    return True

def is_valid_image(image_path: str) -> bool:
    """
    Check if an image file is readable using OpenCV.

    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if the image is readable, False otherwise.
    """
    try:
        return cv2.imread(image_path) is not None
    except Exception:
        return False

def main():
    """
    Main script execution.
    This function parses command-line arguments, validates input/output paths,
    checks for valid images, and calls the rotation prediction function.
    """
    start_time = time.time()
    args = parse_arguments()
    
    # Use platform-independent path resolution
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    model_path = project_root / "models" / "rotation_model.h5"
    
    # Check if model path exists, otherwise look for alternative names
    if not model_path.exists():
        # Try alternative model names
        alternative_paths = [
            project_root / "models" / "label_rotation_model.h5",
            project_root / "models" / "rotation_classifier.h5"
        ]
        for alt_path in alternative_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        else:
            print(f"Error: Rotation model not found. Tried:")
            print(f"  - {project_root / 'models' / 'rotation_model.h5'}")
            for alt_path in alternative_paths:
                print(f"  - {alt_path}")
            print("Please ensure the rotation model is available in the models directory.")

    # Validate input/output paths and retrieve valid images
    valid_images = validate_paths(args.input_image_dir, args.output_image_dir)

    if not valid_images or not validate_model_path(str(model_path)):
        sys.exit(1)

    # Filter out unreadable images
    valid_images = [img for img in valid_images if is_valid_image(img)]
    if not valid_images:
        print("Error: No valid images found for processing.")
        sys.exit(1)

    try:
        predict_angles(args.input_image_dir, args.output_image_dir, str(model_path))
        print(f"\nThe rotated images have been successfully saved in {args.output_image_dir}")
    except Exception as e:
        print(f"Error during image rotation: {e}")
        sys.exit(1)

    print(f"Finished in {round(time.perf_counter() - start_time, 2)} seconds.")

if __name__ == "__main__":
    main()
