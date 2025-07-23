# Import third-party libraries
import unittest
import cv2
import numpy as np
import os
from pathlib import Path

# Import the necessary module from the 'label_processing' module package
from label_processing.text_recognition import ImageProcessor, Tesseract, Threshmode

class TestImageProcessor(unittest.TestCase):
    """
    A test suite for the ImageProcessor class, which handles image processing tasks
    such as loading, preprocessing, and saving images.

    Attributes:
        image_path (Path): The file path to a sample image used for testing.
        image (numpy.ndarray): The image loaded using OpenCV for testing.
    """
    # Ensure image_path is absolute to avoid confusion in CI/CD environments
    image_path = os.path.join(os.path.dirname(__file__), "../testdata/not_empty/coll.mfn-berlin.de_u_1b0791__Crambidae_sp_label_typed_2.jpg")

    # Attempt to load image using OpenCV
    image = cv2.imread(str(image_path))

    def setUp(self):
        """
        Setup for the test case.

        This method is called before each test method. It ensures that the test image
        is loaded correctly. If the image cannot be loaded, a ValueError will be raised
        to alert the user to a missing or invalid image file.

        Raises:
            ValueError: If the image fails to load (None).
        """
        # Print the current working directory to help debug CI/CD issues
        print(f"Current working directory: {os.getcwd()}")
        
        # Print the absolute path of the image file
        print(f"Attempting to load image from: {self.image_path}")
        print(f"Resolved image path: {self.image_path}")

        # Verify that the image exists at the specified path
        if not os.path.exists(self.image_path):
            raise ValueError(f"Image file does not exist at path: {self.image_path}")
        
        # If the image fails to load, raise an exception
        if self.image is None:
            raise ValueError(f"Failed to load image at {self.image_path}")
        else:
            print(f"Image loaded successfully: {self.image_path}")

    def test_construcor_image_from_image(self):
        """
        Test the constructor of ImageProcessor when an image is provided directly.

        This test ensures that the ImageProcessor is correctly initialized with the given
        image and its associated path. It also checks that the image is of the expected type
        (numpy.ndarray) and that the path matches the provided one.

        Asserts:
            IsInstance: Asserts that the image is a numpy.ndarray.
            Equal: Asserts that the path matches the provided image path.
        """
        preprocessor = ImageProcessor(self.image, self.image_path)
        self.assertIsInstance(preprocessor.image, np.ndarray)
        self.assertEqual(Path(preprocessor.path), Path(self.image_path))  # Convert to Path for comparison

    def test_image_processor_from_path(self):
        """
        Test creating an ImageProcessor instance from an image file path.

        This test checks that the ImageProcessor can be initialized by providing the path
        to the image file, ensuring that the image is loaded as a numpy.ndarray and that
        the correct path is assigned to the instance.

        Asserts:
            IsInstance: Asserts that the image is a numpy.ndarray.
            Equal: Asserts that the path matches the expected image path.
        """
        preprocessor = ImageProcessor.read_image(self.image_path)
        self.assertIsInstance(preprocessor.image, np.ndarray)
        self.assertEqual(Path(preprocessor.path), Path(self.image_path))  # Convert to Path for comparison

    def test_image_processor_preprocessing(self):
        """
        Test the preprocessing method of ImageProcessor.

        This test checks that the preprocessing method works correctly with a specified
        threshold mode (e.g., OTSU thresholding). It verifies that the processed image
        is a numpy.ndarray after the operation.

        Asserts:
            IsInstance: Asserts that the preprocessed image is a numpy.ndarray.
        """
        preprocessor = ImageProcessor.read_image(self.image_path)
        preprocessor = preprocessor.preprocessing(Threshmode.OTSU)
        self.assertIsInstance(preprocessor.image, np.ndarray)

    def test_different_pictures(self):
        """
        Test if the preprocessed picture differs from the original image.

        This test checks that the preprocessing step alters the image such that the
        preprocessed image is not identical to the original. It ensures that the image
        has been modified during preprocessing.

        Asserts:
            False: Asserts that the original image and preprocessed image are not identical.
        """
        preprocessor = ImageProcessor.read_image(self.image_path)
        preprocessor = preprocessor.preprocessing(Threshmode.OTSU)
        if self.image.shape == preprocessor.image.shape:
            self.assertFalse(np.allclose(self.image, preprocessor.image))

    def test_save_image(self):
        """
        Test the save_image method of ImageProcessor.
        """
        preprocessor = ImageProcessor.read_image(self.image_path)
        
        # Use absolute path to ensure the output directory exists
        path = os.path.join(os.path.dirname(__file__), "../testdata/output")
        
        # Ensure the output directory exists or create it
        os.makedirs(path, exist_ok=True)
        
        # Convert image_path to a Path object
        image_path_obj = Path(self.image_path)
        
        # Use the name attribute of the Path object
        expected_path = Path(f"{path}/{image_path_obj.name}")
        
        # Print expected path for debugging
        print(f"Expected path: {expected_path}")
        
        preprocessor.save_image(path)
        
        # Verify that the image was saved correctly
        self.assertTrue(expected_path.exists())

    def test_qr_code_reader(self):
        """
        Test the QR code reading method of ImageProcessor.

        This test ensures that the read_qr_code method behaves as expected. If a QR code
        is present in the image, it will return a string with the QR code's value. If
        no QR code is found, it should return None.

        Asserts:
            True: Asserts that the QR code reader returns either None or a string value.
        """
        preprocessor = ImageProcessor.read_image(self.image_path)
        value = preprocessor.read_qr_code()
        self.assertTrue(value == None or isinstance(value, str))


class TestTesseract(unittest.TestCase):
    """
    A test suite for the Tesseract class, which integrates with OCR (Optical Character Recognition)
    to extract text from images.

    Attributes:
        image_path (Path): The file path to a sample image used for testing.
        image (numpy.ndarray): The image loaded using OpenCV for testing.
    """
    image_path = os.path.join(os.path.dirname(__file__), "../testdata/not_empty/coll.mfn-berlin.de_u_1b0791__Crambidae_sp_label_typed_2.jpg")
    image = cv2.imread(str(image_path))

    def setUp(self):
        """
        Setup for the test case.

        This method is called before each test method to ensure the image is loaded
        correctly. If the image cannot be loaded, a ValueError will be raised, which
        alerts the user to a missing or invalid image file.

        Raises:
            ValueError: If the image fails to load (None).
        """
        # Print the current working directory to help debug CI/CD issues
        print(f"Current working directory: {os.getcwd()}")
        
        # Print the absolute path of the image file
        print(f"Attempting to load image from: {self.image_path}")
        print(f"Resolved image path: {self.image_path}")

        # Verify that the image exists at the specified path
        if not os.path.exists(self.image_path):
            raise ValueError(f"Image file does not exist at path: {self.image_path}")
        
        # If the image fails to load, raise an exception
        if self.image is None:
            raise ValueError(f"Failed to load image at {self.image_path}")
        else:
            print(f"Image loaded successfully: {self.image_path}")

    def test_constructor_no_image(self):
        """
        Test the constructor of Tesseract without providing an image.

        This test ensures that the Tesseract instance is properly initialized even if
        no image is provided initially. The image is then assigned to the instance
        after initialization. The test checks that the image is correctly assigned.

        Asserts:
            IsInstance: Asserts that the image is a numpy.ndarray after assignment.
        """
        tesseract_wrapper = Tesseract()
        self.assertIsNone(tesseract_wrapper.image)
        tesseract_wrapper.image = self.image
        self.assertIsInstance(tesseract_wrapper.image, np.ndarray)

    def test_constructor_image(self):
        """
        Test the constructor of Tesseract with an image provided.

        This test checks that the Tesseract instance can be properly initialized with
        an image passed directly during construction. It ensures that the image is
        correctly assigned and is of type numpy.ndarray.

        Asserts:
            IsInstance: Asserts that the image passed to the constructor is a numpy.ndarray.
        """
        tesseract_wrapper = Tesseract(image=self.image)
        self.assertIsInstance(tesseract_wrapper.image, np.ndarray)

    def test_image_to_string(self):
        """
        Test the image_to_string method of Tesseract.

        This test checks that the image_to_string method works correctly by extracting
        text from the image processed by the ImageProcessor. It ensures that the extracted
        text is a string.

        Asserts:
            IsInstance: Asserts that the extracted text is of type string.
        """
        preprocessor = ImageProcessor(self.image, self.image_path)
        tesseract_wrapper = Tesseract(image=preprocessor)
        result = tesseract_wrapper.image_to_string()
        self.assertIsInstance(result["text"], str)
