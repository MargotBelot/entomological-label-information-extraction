# Import third-party libraries
import glob
import os
from PIL import Image
import shutil


def detect_dark_pixels(image: Image, crop_box: tuple, threshold: int = 100) -> float:
    """
    Detect the proportion of dark pixels in an image.

    Args:
        image (Image): Input image.
        crop_box (tuple): (left, upper, right, lower) coordinates for image cropping.
        threshold (int): Threshold for classifying dark pixels. Defaults to 100.

    Returns:
        float: Proportion of dark pixels.
    """
    black_pixels = 0
    total_pixels = 0
    for pixel in image.crop(crop_box).getdata():
        total_pixels += 1
        brightness = sum(pixel) / 3
        if brightness < threshold:
            black_pixels += 1
    return black_pixels / total_pixels


def is_empty(image: Image, crop_margin: float, threshold: float) -> bool:
    """
    Determines if an image is empty based on a given threshold and crop margin.

    Args:
        image: PIL Image object
        crop_margin: float, proportion of the image size to crop from the borders
        threshold: float, proportion of black pixels below which the image is considered empty

    Returns:
        bool, whether the image is empty or not
    """
    width, height = image.size
    crop_box = (
        int(width * crop_margin),
        int(height * crop_margin),
        width - int(width * crop_margin),
        height - int(height * crop_margin),
    )
    return detect_dark_pixels(image, crop_box) < threshold


def find_empty_labels(
    input_folder: str,
    output_folder: str,
    threshold: float = 0.01,
    crop_margin: float = 0.1,
) -> None:
    """
    Find and move empty and non-empty labels to respective folders.

    Args:
        input_folder (str): Path to the directory containing input images.
        output_folder (str): Path to the directory where filtered images will be stored.
        threshold (float): Threshold for classifying empty labels. Defaults to 0.01.
        crop_margin (float): Margin for cropping images. Defaults to 0.1.

    Returns:
        None
    """
    empty_folder = os.path.join(output_folder, "empty")
    not_empty_folder = os.path.join(output_folder, "not_empty")
    os.makedirs(empty_folder, exist_ok=True)
    os.makedirs(not_empty_folder, exist_ok=True)

    for filename in glob.iglob(os.path.join(input_folder, "*")):
        if os.path.isfile(filename):
            try:
                with Image.open(filename) as img:
                    if is_empty(img, crop_margin, threshold):
                        shutil.move(
                            filename,
                            os.path.join(empty_folder, os.path.basename(filename)),
                        )
                    else:
                        shutil.move(
                            filename,
                            os.path.join(not_empty_folder, os.path.basename(filename)),
                        )
            except Exception as e:
                print(f"Error processing {filename}: {e}")
