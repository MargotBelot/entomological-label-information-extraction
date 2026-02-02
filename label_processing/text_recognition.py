# Import third-party libraries
from __future__ import annotations
import os
import copy
import cv2
import shutil
import math
import pytesseract as py
import numpy as np
from typing import Union, Tuple, Optional
from deskew import determine_skew
from enum import Enum
from pathlib import Path
import warnings

# Import the necessary module from the 'label_processing' module package
from label_processing import utils

# Suppress warning messages during execution
warnings.filterwarnings("ignore")

# Constants
CONFIG = r"--psm 6 --oem 3"  # Configuration for OCR
LANGUAGES = "eng+deu+fra+ita+spa+por"  # Specifying languages used for OCR
MIN_SKEW_ANGLE = -10
MAX_SKEW_ANGLE = 10


def find_tesseract() -> None:
    """
    Searches for the tesseract executable and raises an error if it is not found.
    """
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        raise FileNotFoundError(
            (
                "Could not find tesseract on your machine!"
                "Please read the README for instructions!"
            )
        )
    else:
        py.pytesseract.tesseract_cmd = tesseract_path


# ---------------------Image Preprocessing---------------------#


class ImageProcessor:
    """
    A class for image preprocessing and other image actions.
    """

    def __init__(
        self, image: np.ndarray, path: str, blocksize: int = None, c_value: int = None
    ):
        """
        Initialize an instance of Image class.

        Args:
            image (np.ndarray): The image data as a NumPy array.
            path (str): The path to the image file.
            blocksize (int, optional): The blocksize for thresholding. Defaults to None.
            c_value (int, optional): The c_value for thresholding. Defaults to None.
        """
        self.image = image
        self.path = Path(path)
        self.filename = self.path.name
        self.blocksize: Optional[int] = blocksize
        self.c_value: Optional[int] = c_value

    @property
    def blocksize(self) -> int:
        return self._blocksize

    @blocksize.setter
    def blocksize(self, value: int | None) -> None:
        if value is not None:
            if value <= 1 or value % 2 == 0:
                raise ValueError(
                    "Value for blocksize has to be at least 3 and needs\
                    to be odd"
                )
        self._blocksize = value

    @property
    def c_value(self) -> int:
        return self._c_value

    @c_value.setter
    def c_value(self, value: int) -> None:
        self._c_value = value

    @property
    def image(self) -> np.ndarray:
        return self._image

    @image.setter
    def image(self, image: np.ndarray) -> None:
        self._image = image

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        self._path = path

    def copy_this(self) -> ImageProcessor:
        """
        Creates a copy of the current Image instance.

        Returns:
            ImageProcessor: A copy of the current Image instance.
        """
        return copy.copy(self)

    @staticmethod
    def read_image(path: str | Path) -> ImageProcessor:
        """
        Read an image from the specified path and return an instance of the Image class.

        Args:
            path (str): The path to a JPG file.

        Returns:
            Image: An instance of the Image class.
        """
        return ImageProcessor(cv2.imread(str(path)), path)

    def get_grayscale(self) -> ImageProcessor:
        """
        Convert the image to grayscale.

        Returns:
            Image: An instance of the Image class representing the grayscale image.
        """
        image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def blur(self, ksize: tuple[int, int] = (5, 5)) -> ImageProcessor:
        """
        Apply Gaussian blur to the image.

        Args:
            ksize (Tuple[int, int], optional): The kernel size for blurring. Defaults to (5, 5).

        Returns:
            Image: An instance of the Image class representing the blurred image.
        """
        image = cv2.GaussianBlur(self.image, ksize, 0)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def remove_noise(self) -> ImageProcessor:
        """
        Remove noise from the image using median blur.

        Returns:
            Image: An instance of the Image class representing the noise-reduced image.
        """
        image = cv2.medianBlur(self.image, 5)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def apply_clahe(
        self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
    ) -> ImageProcessor:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

        CLAHE improves contrast in images with uneven illumination or low contrast,
        which is common in aged specimen labels or images with inconsistent lighting.

        Args:
            clip_limit (float, optional): Threshold for contrast limiting. Higher values
                give more contrast. Defaults to 2.0.
            tile_grid_size (tuple[int, int], optional): Size of grid for histogram equalization.
                Defaults to (8, 8).

        Returns:
            ImageProcessor: An instance of the Image class with CLAHE applied.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        image = clahe.apply(self.image)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def normalize_illumination(self) -> ImageProcessor:
        """
        Normalize image illumination using morphological operations.

        This method corrects uneven lighting by estimating and removing the background
        illumination, useful for images with shadows or uneven flash lighting.

        Returns:
            ImageProcessor: An instance of the Image class with normalized illumination.
        """
        # Estimate background using morphological closing with large kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        background = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)

        # Subtract background and normalize to 0-255 range
        image = cv2.subtract(background, self.image)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def thresholding(self, thresh_mode: Enum) -> ImageProcessor:
        """
        Perform thresholding on the image.

        Args:
            thresh_mode (Threshmode): The thresholding mode to use (OTSU, ADAPTIVE_MEAN, or ADAPTIVE_GAUSSIAN).

        Returns:
            Image: An instance of the Image class representing the thresholded image.
        """
        if thresh_mode == Threshmode.OTSU:
            image = cv2.threshold(
                self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        elif thresh_mode == Threshmode.ADAPTIVE_GAUSSIAN:
            # set blocksize and c_value
            gaussian_blocksize = self.blocksize if self.blocksize else 73
            gaussian_c = self.c_value if self.c_value else 16

            image = cv2.adaptiveThreshold(
                self.image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                gaussian_blocksize,
                gaussian_c,
            )
        elif thresh_mode == Threshmode.ADAPTIVE_MEAN:
            # set blocksize and c_value
            mean_blocksize = self.blocksize if self.blocksize else 35
            mean_c = self.c_value if self.c_value else 17

            image = cv2.adaptiveThreshold(
                self.image,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                mean_blocksize,
                mean_c,
            )
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def dilate(self) -> ImageProcessor:
        """
        Dilate the image using a 5x5 kernel.

        Returns:
            Image: An instance of the Image class representing the dilated image.
        """
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.dilate(self.image, kernel, iterations=1)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def erode(self) -> ImageProcessor:
        """
        Erode the image using a 5x5 kernel.

        Returns:
            Image: An instance of the Image class representing the eroded image.
        """
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.erode(self.image, kernel, iterations=1)
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    @staticmethod
    def _rotate(
        image: np.ndarray,
        angle: float | np.float,
        background: Union[int, Tuple[int, int, int]],
    ) -> np.ndarray:
        """
        Performs a rotation of an image given an angle.

        Args:
            image (np.ndarray): Image loaded in with OpenCV.
            angle (float): Angle with which the picture should be rotated.
            background (Union[int, Tuple[int, int, int]]): RGB values for the background color.

        Returns:
            np.ndarray: Rotated image.
        """
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(
            np.cos(angle_radian) * old_width
        )
        height = abs(np.sin(angle_radian) * old_width) + abs(
            np.cos(angle_radian) * old_height
        )

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(
            image,
            rot_mat,
            (int(round(height)), int(round(width))),
            borderValue=background,
        )

    def get_skew_angle(self) -> Optional[np.float64]:
        """
        Calculate and return the skew angle of the image.

        Returns:
            Optional[np.float64]: The skew angle in degrees or None if it couldn't be determined.
        """
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(
            grayscale, max_angle=MAX_SKEW_ANGLE, min_angle=MIN_SKEW_ANGLE
        )
        return angle

    def deskew(self, angle: Optional[np.float64]) -> ImageProcessor:
        """
        Rotate the image to deskew it.

        Args:
            angle (Optional[np.float64]): The skew angle to use for deskewing.

        Returns:
            Image: An instance of the Image class representing the deskewed image.
        """
        if angle is None:
            # Handle the case where angle is None, e.g., log a message or skip deskewing
            print(
                f"Warning: Skew angle for file {self.filename} could not be determined. Skipping deskewing."
            )
            return self

        # If angle is not None, proceed with deskewing
        image = self._rotate(self.image, angle, (255, 255, 255))
        image_instance = self.copy_this()
        image_instance.image = image
        return image_instance

    def preprocessing(
        self,
        thresh_mode: Threshmode,
        use_clahe: bool = False,
        normalize_illum: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
    ) -> ImageProcessor:
        """
        Perform a series of preprocessing steps on the image.

        Args:
            thresh_mode (Threshmode): The thresholding mode to use (OTSU, ADAPTIVE_MEAN, or ADAPTIVE_GAUSSIAN).
            use_clahe (bool, optional): Apply CLAHE for contrast enhancement. Useful for low-contrast
                or faded labels. Defaults to False.
            normalize_illum (bool, optional): Apply illumination normalization to correct uneven lighting.
                Useful for images with shadows or hotspots. Defaults to False.
            clahe_clip_limit (float, optional): CLAHE contrast limiting threshold. Defaults to 2.0.
            clahe_tile_grid_size (tuple[int, int], optional): CLAHE grid size. Defaults to (8, 8).

        Returns:
            ImageProcessor: An instance of the Image class representing the preprocessed image.
        """
        # Skew angle has to be calculated before processing
        angle = self.get_skew_angle()

        if angle is None:
            print(
                "Warning: Skew angle could not be determined. Skipping preprocessing."
            )
            return self

        # Perform preprocessing
        image = self.get_grayscale()

        # Optional: normalize illumination before other processing
        if normalize_illum:
            image = image.normalize_illumination()

        # Optional: apply CLAHE before blurring for better contrast
        if use_clahe:
            image = image.apply_clahe(
                clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size
            )

        image = image.blur()
        image = image.thresholding(thresh_mode=thresh_mode)
        image = image.deskew(angle)
        return image

    # ---------------------Read QR-Code---------------------#

    def read_qr_code(self) -> Optional[str]:
        """
        Tries to identify if a picture has a QR-code and then reads and returns it.

        Returns:
            Optional[str]: Decoded QR-code text as a str or None if there is no QR-code found.
        """
        try:
            detect = cv2.QRCodeDetector()
            value = detect.detectAndDecode(self.image)[0]
            return value if value else None
        except Exception as e:
            print(f"An error occurred while detecting and decoding QR code: {e}")
            return None

    def save_image(self, dir_path: str | Path, appendix: Optional[str] = None) -> None:
        """
        Save the image to a specified directory with an optional appendix.

        Args:
            dir_path (str | Path): The directory path where the image will be saved.
            appendix (str, optional): An optional string to append to the image filename. Defaults to None.
        """
        try:
            if appendix:
                filename = utils.generate_filename(
                    self.filename, appendix, extension="jpg"
                )
            else:
                filename = self.filename
            filename_processed = os.path.join(dir_path, filename)
            cv2.imwrite(filename_processed, self.image)
        except Exception as e:
            print(f"An error occurred while saving the image: {e}")


class Threshmode(Enum):
    """
    Different possibilities for thresholding.

    Args:
        Enum (int):
    """

    OTSU = 1
    ADAPTIVE_MEAN = 2
    ADAPTIVE_GAUSSIAN = 3

    @classmethod
    def eval(cls, threshmode: int) -> Enum:
        if threshmode == 1:
            return cls.OTSU
        if threshmode == 2:
            return cls.ADAPTIVE_MEAN
        if threshmode == 3:
            return cls.ADAPTIVE_GAUSSIAN


# ---------------------OCR Tesseract---------------------#


class Tesseract:
    def __init__(
        self, languages=LANGUAGES, config=CONFIG, image: Optional[ImageProcessor] = None
    ) -> None:
        """
        Initialize the Tesseract OCR processor.

        Args:
            languages (str, optional): OCR available languages. Defaults to LANGUAGES.
            config (str, optional): Additional custom configuration flags not available via the pytesseract function. Defaults to CONFIG.
            image (ImageProcessor, optional): An instance of the Image class representing the image to process. Defaults to None.
        """
        self.config = config
        self.languages = languages
        self.image = image

    @property
    def image(self) -> ImageProcessor:
        return self._image

    @image.setter
    def image(self, img: ImageProcessor) -> None:
        self._image = img

    @staticmethod
    def _process_string(result_raw: str) -> str:
        """
        Processes the OCR output by replacing '\n' with spaces and encoding it to ASCII and decoding it again to UTF-8.

        Args:
            result_raw (str): Raw string from pytesseract output.

        Returns:
            str: Processed string.
        """
        processed = result_raw.replace("\n", " ")
        return processed

    def image_to_string(self) -> dict[str, str | float]:
        """
        Apply OCR and image parameters on JPG images.

        Returns:
            dict[str, str | float]: A dictionary containing the image ID (filename), OCR-processed text, and confidence score.
        """
        # Get OCR text with confidence data
        data = py.image_to_data(
            self.image.image,
            lang=self.languages,
            config=self.config,
            output_type=py.Output.DICT,
        )

        # Extract text and calculate mean confidence
        text_parts = []
        confidences = []

        for i in range(len(data["text"])):
            if int(data["conf"][i]) > 0:  # Only include words with positive confidence
                text_parts.append(data["text"][i])
                confidences.append(int(data["conf"][i]))

        # Combine text and calculate average confidence
        transcript = " ".join(text_parts)
        transcript = self._process_string(transcript)

        # Calculate mean confidence (0-100 scale, convert to 0-1)
        mean_confidence = (
            (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
        )

        return {
            "ID": self.image.filename,
            "text": transcript,
            "confidence": round(mean_confidence, 3),
        }
