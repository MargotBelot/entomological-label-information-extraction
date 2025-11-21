# Import third-party libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import warnings
from typing import List
import shutil
import logging
from PIL import Image


warnings.filterwarnings("ignore", category=UserWarning, module="absl")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define constants
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4
ANGLE_MAP = {0: 0, 1: 90, 2: 180, 3: 270}

# ------------------ Image Loading & Processing ------------------ #


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image.
    """
    # Convert Path object to string if necessary
    image_path_str = str(image_path)
    image = cv2.imread(image_path_str)
    if image is None:
        raise ValueError(f"Error: Unable to read image '{image_path}'")
    return image


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate an image based on a given angle.

    Args:
        image (np.ndarray): Input image.
        angle (int): Angle of rotation in multiples of 90 degrees.

    Returns:
        np.ndarray: Rotated image.
    """
    height, width = image.shape[:2]
    target_angle = (4 - angle) % NUM_CLASSES
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), target_angle * 90, 1
    )

    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])

    new_width = int(height * sin_theta + width * cos_theta)
    new_height = int(height * cos_theta + width * sin_theta)

    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    return cv2.warpAffine(image, rotation_matrix, (new_width, new_height))


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to a file path.

    Args:
        image (np.ndarray): Image to save.
        output_path (str): Path to save the image.

    Returns:
        bool: True if the image is saved, False otherwise.
    """
    return cv2.imwrite(output_path, image)


# ------------------ Image Rotation ------------------ #


def rotate_single_image(image_path: str, angle: int, output_dir: str) -> bool:
    """
    Rotate a single image based on a given angle and save the rotated image.

    Args:
        image_path (str): Path to the input image file.
        angle (int): Angle of rotation in multiples of 90 degrees.
        output_dir (str): Directory to save the rotated image.

    Returns:
        bool: True if the image is rotated, False otherwise.
    """
    try:
        image = load_image(image_path)
        if angle == 0:
            print(f"Skipping image '{image_path}' as it does not need rotation.")
            return save_image(
                image, os.path.join(output_dir, os.path.basename(image_path))
            )

        rotated_image = rotate_image(image, angle)
        output_path = os.path.join(output_dir, os.path.basename(image_path))

        if save_image(rotated_image, output_path):
            print(
                f"Successfully rotated image '{image_path}' by {angle * 90} degrees to reach 0 degree."
            )
            return True
        else:
            print(f"Error: Failed to write rotated image '{image_path}' to file.")
            return False
    except Exception as e:
        print(
            f"Error: An exception occurred while processing image '{image_path}': {e}"
        )
        return False


# ------------------ Model Prediction & Processing ------------------ #


def get_image_paths(input_image_dir: str) -> List[str]:
    """
    Get a list of image paths in the input directory.

    Args:
        input_image_dir (str): Directory containing input images.

    Returns:
        list: List of image paths.
    """
    return [
        os.path.join(input_image_dir, filename)
        for filename in os.listdir(input_image_dir)
        if filename.lower().endswith((".jpg", ".jpeg", ".tiff", ".tif"))
    ]


def load_images(image_paths: List[str]) -> np.ndarray:
    """
    Load images from a list of image paths.

    Args:
        image_paths (list): List of image paths.

    Returns:
        np.ndarray: Loaded images.
    """
    images = []
    for image_path in image_paths:
        image = load_image(image_path)
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
    return np.array(images, dtype=np.float32) / 255.0


def get_predicted_angles(model: tf.keras.Model, images: np.ndarray) -> List[int]:
    """
    Predict angles for a list of images using a trained model.

    Args:
        model (tf.keras.Model): Trained model.
        images (np.ndarray): List of images.

    Returns:
        list: List of predicted angles.
    """
    predictions = model.predict(images)
    return np.argmax(predictions, axis=1)


def rotate_images(
    image_paths: List[str], predicted_angles: List[int], output_image_dir: str
) -> None:
    """
    Rotate images based on their predicted angles and save them to the output directory.

    Args:
        image_paths (list): List of image paths.
        predicted_angles (list): List of predicted angles.
        output_image_dir (str): Directory to save rotated images.

    Returns:
        None
    """
    num_rotated = 0
    num_skipped = 0
    for image_path, predicted_angle in zip(image_paths, predicted_angles):
        if rotate_single_image(image_path, predicted_angle, output_image_dir):
            num_rotated += 1
        else:
            num_skipped += 1
    print(f"Total images rotated: {num_rotated}")
    print(f"Total images skipped: {num_skipped}")


# ------------------ Debugging Function ------------------ #


def debug_save_by_angle(image_paths, predicted_angles, output_base_dir):
    angle_names = {0: "0", 1: "90", 2: "180", 3: "270"}
    for img_path, angle in zip(image_paths, predicted_angles):
        angle_folder = os.path.join(output_base_dir, angle_names.get(angle, "unknown"))
        os.makedirs(angle_folder, exist_ok=True)
        output_path = os.path.join(angle_folder, os.path.basename(img_path))
        shutil.copy2(img_path, output_path)


# ------------------ Main Function to Predict & Rotate ------------------ #


def predict_angles(
    input_image_dir: str, output_image_dir: str, model_path: str, debug: bool = False
) -> None:
    """
    Load a trained model, predict angles for input images, and rotate images accordingly.

    Args:
        input_image_dir (str): Directory containing input images.
        output_image_dir (str): Directory to save rotated images.
        model_path (str): Path to the trained model.
        debug (bool): If True, saves images by predicted angles for debugging.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ANGLE_MAP = {0: 0, 1: 90, 2: 180, 3: 270}

    os.makedirs(output_image_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    if not os.path.exists(input_image_dir):
        print(f"Error: Input directory '{input_image_dir}' not found.")
        return

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    print("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    image_paths = get_image_paths(input_image_dir)
    if not image_paths:
        print("No images found in the input directory.")
        return

    images = load_images(image_paths)
    print("Predicting angles...")
    predicted_angles = get_predicted_angles(model, images)
    print("Predicted classes:", predicted_angles)

    # Diagnostic: print angle counts
    angle_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for angle in predicted_angles:
        angle_counts[angle] += 1
    logging.info(f"Angle prediction counts: {angle_counts}")

    if debug:
        debug_save_by_angle(
            image_paths, predicted_angles, output_base_dir="debug_angles"
        )

    print("Rotating images based on predictions...")

    # Write rotation metadata for consolidation
    import csv

    meta_path = os.path.join(output_image_dir, "rotation_metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle", "corrected"])

        for image_path, predicted_angle in zip(image_paths, predicted_angles):
            angle_deg = ANGLE_MAP.get(predicted_angle, 0)
            filename = os.path.basename(image_path)
            corrected = bool(angle_deg != 0)

            # Write metadata
            writer.writerow([filename, angle_deg, corrected])

            # Rotate the image
            rotate_single_image(
                image_path, angle_deg // 90, output_image_dir
            )  # if your function expects multiples of 90


def rotate_image_pil(image_path, angle_deg, output_path):
    with Image.open(image_path) as img:
        rotated = img.rotate(angle_deg, expand=True)
        rotated.save(output_path)
