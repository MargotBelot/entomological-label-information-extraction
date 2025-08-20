# Third-Party Libraries
import argparse
import os
from glob import glob
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.models import load_model
from keras.layers import BatchNormalization
import time

# Constants
IMAGE_SIZE = (224, 224)
TEXT_FILE = "accuracy_metrics.txt"
ANGLE_NAMES = ['0', '90', '180', '270']
NUM_CLASSES = 4


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    usage = 'rotation_eval.py [-h] -i <input image dir> -o <output folder path>'

    parser = argparse.ArgumentParser(
        description="Create and save rotation evaluation metrics.",
        add_help = False,
        usage = usage)

    parser.add_argument(
            '-h','--help',
            action='help',
            help='Open this help text.'
            )
    
    parser.add_argument(
            '-i', '--input_image_dir',
            metavar='',
            type=str,
            required = True,
            help=('Path to the image input folder.')
            )
            
    parser.add_argument(
            '-o', '--output_folder_path',
            metavar='',
            type=str,
            default = os.getcwd(),
            help=('Path to the output folder.')
            )

    return parser.parse_args()

def load_images(input_image_dir: str) -> tuple:
    """
    Load images from the given directory and extract ground truth labels.
    
    Args:
        input_image_dir (str): Directory containing images.
    
    Returns:
        tuple: (Loaded images as numpy array, Ground truth labels as numpy array, List of filenames)
    """
    true_labels = []
    loaded_images = []
    filenames = []
    for img_path in glob(os.path.join(input_image_dir, '*.jpg')):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image '{img_path}'. Skipping.")
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        loaded_images.append(img)
        filenames.append(img_path)
        angle = int(img_path.split('__')[-1].split('.')[0]) // 90
        true_labels.append(angle)
    return np.array(loaded_images), np.array(true_labels), filenames

def rotate_image(img_path: str, angle: int) -> None:
    """
    Rotate the image by the given angle and save it back to the same path.
    
    Args:
        img_path (str): Path to the image file.
        angle (int): Rotation angle index (0, 1, 2, 3 corresponding to 0, 90, 180, 270 degrees).
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to read image '{img_path}'.")
            return
        if angle == 0:
            return
        height, width = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), (4 - angle) % NUM_CLASSES * 90, 1)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
        cv2.imwrite(img_path, rotated_img)
    except Exception as e:
        print(f"Error rotating image '{img_path}': {e}")

def evaluate_rotation_model(input_image_dir: str, output_folder_path: str) -> None:
    """
    Load model, predict rotations, and evaluate performance.
    
    Args:
        input_image_dir (str): Directory containing images.
        output_folder_path (str): Path to save evaluation results.
    """
    start_time = time.time()
    images, true_labels, filenames = load_images(input_image_dir)
    if len(images) == 0:
        print("Error: No valid images found.")
        return
    
    # Use platform-independent path resolution
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    model_path = project_root / "models" / "rotation_model.h5"
    
    # Check if model exists, otherwise try alternative names
    if not model_path.exists():
        alternative_paths = [
            project_root / "models" / "label_rotation_model.h5",
            project_root / "models" / "rotation_classifier.h5"
        ]
        for alt_path in alternative_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        else:
            print(f"Error: Rotation model not found at {model_path}")
            return
    
    try:
        model = load_model(str(model_path), custom_objects={"BatchNormalization": BatchNormalization})
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    predicted_labels = np.argmax(model.predict(images), axis=1)
    for img_path, predicted_angle in zip(filenames, predicted_labels):
        rotate_image(img_path, predicted_angle)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    os.makedirs(output_folder_path, exist_ok=True)
    accuracy_file_path = os.path.join(output_folder_path, TEXT_FILE)
    with open(accuracy_file_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-score: {f1:.2f}\n")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xticks(ticks=np.arange(4) + 0.5, labels=ANGLE_NAMES)
    plt.yticks(ticks=np.arange(4) + 0.5, labels=ANGLE_NAMES)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_folder_path, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Finished in {round(time.time() - start_time, 2)} seconds")

def main():
    """
    Main function to execute rotation model evaluation.
    """
    args = parse_arguments()
    evaluate_rotation_model(args.input_image_dir, args.output_folder_path)

if __name__ == "__main__":
    main()
