# Import third-party libraries
import unittest
import numpy as np
import os

# Import the necessary module from the 'label_processing' module package
from label_processing.label_rotation import load_image, rotate_image

class TestImageRotationModule(unittest.TestCase):
    """
    A test case for the image rotation functionality.

    This test suite ensures that the functions responsible for loading and rotating images work as expected.
    It verifies that images can be loaded properly and that rotation functions maintain correct data types
    and image dimensions.
    """

    def test_load_image(self):
        """
        Test the load_image function.

        Ensures that the image is loaded successfully from the given path, 
        and that the returned object is a numpy array. Also checks that the 
        image is not empty.
        """
        img_path = os.path.join(os.path.dirname(__file__), "../testdata/not_empty/147843c0-06a7-496c-a156-0b139e843d62_label_front_0001_label_single.jpg")
        
        # Ensure the image can be loaded successfully
        image = load_image(img_path)
        self.assertIsInstance(image, np.ndarray, "Loaded image is not a numpy array!")
        self.assertGreater(image.size, 0, "Loaded image is empty!")

    def test_rotate_image(self):
        """
        Test the rotate_image function.

        Verifies that the image can be rotated by a specified angle, and that the rotated
        image retains its data type (numpy array) and has the expected dimensions (height, width, channels).
        """
        img_path = os.path.join(os.path.dirname(__file__), "../testdata/not_empty/147843c0-06a7-496c-a156-0b139e843d62_label_front_0001_label_single.jpg")
        angle = 1  # Rotate by 1 degree (or any other specified angle)
        
        # Load the image
        image = load_image(img_path)
        
        # Rotate the image
        rotated_image = rotate_image(image, angle)
        
        # Ensure the rotated image is still a numpy array and has a shape
        self.assertIsInstance(rotated_image, np.ndarray, "Rotated image is not a numpy array!")
        self.assertEqual(len(rotated_image.shape), 3, "Rotated image does not have 3 dimensions (height, width, channels)!")

if __name__ == '__main__':
    unittest.main()
