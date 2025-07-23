# Import third-party libraries
import unittest
import shutil
from pathlib import Path
from PIL import Image

# Import the necessary module from the 'label_processing' module package
from label_processing.detect_empty_labels import *

class TestImageFilter(unittest.TestCase):
    """Test suite for image filtering functions in the 'detect_empty_labels' module."""

    def setUp(self):
        """
        Set up the test environment by creating directories and test images.

        This method is called before each test. It creates test directories for input and output, 
        and generates two sample images: one dark (mostly black) and one light (mostly white).
        """
        self.test_input_dir = Path("testdata/empty_test")
        self.test_output_dir = Path("testdata/output")

        # Ensure input and output directories exist
        self.test_input_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Create test images
        self.dark_image_path = self.test_input_dir / "dark.jpg"
        self.light_image_path = self.test_input_dir / "light.jpg"

        # Create a dark image (mostly black)
        dark_image = Image.new("RGB", (100, 100), color=(0, 0, 0))
        dark_image.save(self.dark_image_path)

        # Create a light image (mostly white)
        light_image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        light_image.save(self.light_image_path)

    def tearDown(self):
        """
        Clean up the test environment by removing the created directories and files.

        This method is called after each test to remove the generated test directories and images 
        to ensure a clean state for the next test.
        """
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    def test_detect_dark_pixels(self):
        """
        Test detecting dark pixels in an image.

        This test verifies the detection of dark pixels in images. It checks that the ratio of 
        dark pixels is high for dark images and low for light images.
        """
        dark_image = Image.open(self.dark_image_path)
        light_image = Image.open(self.light_image_path)

        # Test if the ratio of dark pixels is high for dark image and low for light image
        dark_ratio = detect_dark_pixels(dark_image, (0, 0, 100, 100), threshold=100)
        light_ratio = detect_dark_pixels(light_image, (0, 0, 100, 100), threshold=100)

        self.assertGreater(dark_ratio, 0.9)  # Dark image should have a high proportion of dark pixels
        self.assertLess(light_ratio, 0.1)  # Light image should have a low proportion of dark pixels

    def test_is_empty(self):
        """
        Test whether an image is classified as empty.

        This test checks if the is_empty function correctly classifies an image as empty 
        based on its pixel content. A light image (mostly white) is expected to be empty, 
        while a dark image (mostly black) is not.
        """
        dark_image = Image.open(self.dark_image_path)
        light_image = Image.open(self.light_image_path)

        # Test if light image is considered empty (mostly white) and dark image is not empty (dark)
        self.assertFalse(is_empty(dark_image, crop_margin=0.1, threshold=0.9))  # Dark image should not be empty
        self.assertTrue(is_empty(light_image, crop_margin=0.1, threshold=0.9))  # Light image should be empty

    def test_find_empty_labels(self):
        """
        Test the classification of images into empty and not empty folders.

        This test checks the find_empty_labels function to ensure images are correctly categorized
        into 'empty' and 'not_empty' directories based on their content.
        """
        find_empty_labels(str(self.test_input_dir), str(self.test_output_dir), threshold=0.9, crop_margin=0.1)

        empty_folder = self.test_output_dir / "empty"
        not_empty_folder = self.test_output_dir / "not_empty"

        # Check if the light image is in the "empty" folder and dark image in "not_empty"
        self.assertTrue((empty_folder / "light.jpg").exists(), "Light image should be in the 'empty' folder")
        self.assertTrue((not_empty_folder / "dark.jpg").exists(), "Dark image should be in the 'not_empty' folder")

if __name__ == "__main__":
    unittest.main()
