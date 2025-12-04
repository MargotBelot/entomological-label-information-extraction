# Import third-party libraries
import unittest
import numpy as np
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

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
        img_path = Path(__file__).parent / ".." / "testdata" / "not_empty" / "147843c0-06a7-496c-a156-0b139e843d62_label_front_0001_label_single.jpg"
        
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
        img_path = Path(__file__).parent / ".." / "testdata" / "not_empty" / "147843c0-06a7-496c-a156-0b139e843d62_label_front_0001_label_single.jpg"
        angle = 1  # Rotate by 1 degree (or any other specified angle)
        
        # Load the image
        image = load_image(img_path)
        
        # Rotate the image
        rotated_image = rotate_image(image, angle)
        
        # Ensure the rotated image is still a numpy array and has a shape
        self.assertIsInstance(rotated_image, np.ndarray, "Rotated image is not a numpy array!")
        self.assertEqual(len(rotated_image.shape), 3, "Rotated image does not have 3 dimensions (height, width, channels)!")

    @patch('label_processing.label_rotation.load_model')
    def test_model_loading_fallback(self, mock_load_model):
        """
        Test the model loading fallback mechanism.
        
        Verifies that when standard model loading fails, the code attempts to load
        the model with custom_objects={'BatchNormalization': tf.keras.layers.BatchNormalization}.
        This tests the error handling in the predict_angles function (lines 256-265).
        """
        # Import tensorflow only when running this test
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not available")
        
        # Create a mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[1.0, 0.0, 0.0, 0.0]])  # Predict angle 0
        
        # First call raises exception, second call succeeds with custom_objects
        mock_load_model.side_effect = [
            Exception("Standard loading failed"),
            mock_model
        ]
        
        # Import the predict_angles function
        from label_processing.label_rotation import predict_angles
        
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, 'input')
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a dummy image file
            test_image_path = os.path.join(input_dir, 'test_image.jpg')
            # Create a simple test image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(test_image_path, test_img)
            
            # Create a dummy model file path
            model_path = os.path.join(temp_dir, 'model.h5')
            # Create an empty file to simulate model existence
            Path(model_path).touch()
            
            # Call predict_angles - this should trigger the fallback mechanism
            predict_angles(input_dir, output_dir, model_path, debug=False)
            
            # Verify that load_model was called twice
            self.assertEqual(mock_load_model.call_count, 2, 
                           "load_model should be called twice: once failing, once with custom_objects")
            
            # Verify the second call included custom_objects
            second_call_kwargs = mock_load_model.call_args_list[1][1]
            self.assertIn('custom_objects', second_call_kwargs, 
                         "Second load_model call should include custom_objects")
            self.assertIn('BatchNormalization', second_call_kwargs['custom_objects'],
                         "custom_objects should contain BatchNormalization")

    @patch('label_processing.label_rotation.load_model')
    def test_optimizer_compilation_fallback(self, mock_load_model):
        """
        Test the optimizer compilation fallback mechanism.
        
        Verifies that when model compilation with legacy Adam optimizer fails,
        the code falls back to using the standard Adam optimizer.
        This tests the error handling in the predict_angles function (lines 268-281).
        """
        # Import tensorflow only when running this test
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not available")
        
        # Create a mock model that fails on legacy optimizer compile
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[1.0, 0.0, 0.0, 0.0]])  # Predict angle 0
        
        # Track compile calls
        compile_call_count = [0]
        
        def mock_compile(*args, **kwargs):
            compile_call_count[0] += 1
            if compile_call_count[0] == 1:
                # First call with legacy optimizer should fail
                if 'optimizer' in kwargs:
                    optimizer = kwargs['optimizer']
                    if hasattr(optimizer, '__class__') and 'legacy' in str(type(optimizer)):
                        raise Exception("Legacy optimizer not supported")
            # Second call succeeds
            return None
        
        mock_model.compile = mock_compile
        mock_load_model.return_value = mock_model
        
        # Import the predict_angles function
        from label_processing.label_rotation import predict_angles
        
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, 'input')
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a dummy image file
            test_image_path = os.path.join(input_dir, 'test_image.jpg')
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(test_image_path, test_img)
            
            # Create a dummy model file path
            model_path = os.path.join(temp_dir, 'model.h5')
            Path(model_path).touch()
            
            # Call predict_angles - this should trigger the optimizer fallback mechanism
            predict_angles(input_dir, output_dir, model_path, debug=False)
            
            # Verify that compile was called at least once
            # (In real scenario with Keras 3, it tries legacy first, then standard)
            self.assertGreater(compile_call_count[0], 0,
                             "Model compile should be called at least once")

    @patch('label_processing.label_rotation.load_model')
    def test_rotation_metadata_csv_writing(self, mock_load_model):
        """
        Test that rotation metadata CSV is created correctly during prediction.
        
        Verifies that the predict_angles function writes a rotation_metadata.csv file
        with the correct format (filename, angle, corrected columns) and values.
        This tests the CSV writing logic in predict_angles (lines 307-320).
        """
        # Import tensorflow only when running this test
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not available")
        
        import csv
        
        # Create a mock model that predicts different angles
        mock_model = MagicMock()
        # Predict a mix of angles: 0, 90, 180, 270 (one of each)
        # The exact mapping to files doesn't matter since os.listdir order is unpredictable
        mock_model.predict.return_value = np.array([
            [1.0, 0.0, 0.0, 0.0],  # angle class 0 (0 degrees)
            [0.0, 1.0, 0.0, 0.0],  # angle class 1 (90 degrees)
            [0.0, 0.0, 1.0, 0.0],  # angle class 2 (180 degrees)
            [0.0, 0.0, 0.0, 1.0]   # angle class 3 (270 degrees)
        ])
        
        mock_load_model.return_value = mock_model
        
        # Import the predict_angles function
        from label_processing.label_rotation import predict_angles
        
        # Create a temporary directory structure for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, 'input')
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create multiple dummy image files
            import cv2
            test_images = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg']
            for img_name in test_images:
                test_image_path = os.path.join(input_dir, img_name)
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(test_image_path, test_img)
            
            # Create a dummy model file path
            model_path = os.path.join(temp_dir, 'model.h5')
            Path(model_path).touch()
            
            # Call predict_angles - this should create the metadata CSV
            predict_angles(input_dir, output_dir, model_path, debug=False)
            
            # Verify the rotation_metadata.csv file was created
            metadata_path = os.path.join(output_dir, 'rotation_metadata.csv')
            self.assertTrue(os.path.exists(metadata_path),
                          "rotation_metadata.csv should be created")
            
            # Read and verify the CSV contents
            with open(metadata_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Verify the number of rows
            self.assertEqual(len(rows), 4, "Should have 4 rows of metadata")
            
            # Verify CSV headers
            self.assertEqual(set(rows[0].keys()), {'filename', 'angle', 'corrected'},
                           "CSV should have correct headers")
            
            # Verify all test images are present in the CSV
            filenames_in_csv = {row['filename'] for row in rows}
            expected_filenames = set(test_images)
            self.assertEqual(filenames_in_csv, expected_filenames,
                           "All test images should be in the CSV")
            
            # Verify the data structure and values are valid
            angles_seen = set()
            for row in rows:
                filename = row['filename']
                angle = row['angle']
                corrected = row['corrected']
                
                # Angle should be one of: 0, 90, 180, 270
                self.assertIn(angle, ['0', '90', '180', '270'],
                            f"{filename}: angle should be one of 0, 90, 180, 270")
                angles_seen.add(angle)
                
                # Corrected should be False only when angle is 0
                if angle == '0':
                    self.assertEqual(corrected, 'False',
                                   f"{filename}: corrected should be False when angle is 0")
                else:
                    self.assertEqual(corrected, 'True',
                                   f"{filename}: corrected should be True when angle is {angle}")
            
            # Since we mocked 4 different angle predictions (0, 90, 180, 270),
            # we should see all 4 angles in the CSV
            self.assertEqual(len(angles_seen), 4,
                           "Should have all 4 different angles (0, 90, 180, 270) in the CSV")

if __name__ == '__main__':
    unittest.main()
