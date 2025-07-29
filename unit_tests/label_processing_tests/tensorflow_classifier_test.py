# Import third-party libraries
import unittest
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
import os
from tensorflow import keras

# Import the necessary module from the 'label_processing' module package
from label_processing.tensorflow_classifier import *

class TestTFClassifier(unittest.TestCase):
    """
    A test suite for the TensorFlow classifier module.
    """
    model_path = Path(__file__).parent / ".." / ".." / "models" / "label_classifier_hp"
    classes = ['handwritten', 'printed']
    outdir = Path(__file__).parent / ".." / "testdata" / "output"
    jpg_dir = Path(__file__).parent / ".." / "testdata" / "not_empty"
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Ensure output directory exists
        os.makedirs(self.outdir, exist_ok=True)
        
        try:
            # Try to load the model
            if os.path.exists(self.model_path):
                # Set TensorFlow to use CPU only in tests
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                self.model = get_model(self.model_path)
            else:
                self.skipTest(f"Model not found at {self.model_path}")
        except Exception as e:
            self.skipTest(f"Failed to load model: {str(e)}")

    def test_class_prediction_normal(self):
        """
        Test the normal case of class prediction.

        This test checks if the class prediction function creates the expected output CSV file
        and if the number of rows in the DataFrame matches the number of images in the input directory.
        """
        # Test class_prediction with the actual model
        df = class_prediction(self.model, self.classes, self.jpg_dir, self.outdir)

        # Use an absolute path to ensure the file is checked in the correct directory
        output_file = os.path.join(self.outdir, "not_empty_prediction_classifer.csv")
        
        # Check if the output CSV file is generated
        self.assertTrue(os.path.exists(output_file), f"Output file not found: {output_file}")
        
        # Ensure the DataFrame's length matches the number of images in the directory
        self.assertEqual(len(df.index), len(os.listdir(self.jpg_dir)))

    def test_class_prediction_empty(self):
        """
        Test the case of class prediction with an empty directory.

        This test checks if the function raises a FileNotFoundError when given an empty input directory.
        """
        empty_dir = Path(__file__).parent / ".." / "testdata" / "empty_dir"
        Path(empty_dir).mkdir(parents=True, exist_ok=True)

        # Check that FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            class_prediction(self.model, self.classes, empty_dir, self.outdir)

    def test_create_dirs(self):
        """
        Test the creation of directories based on class predictions.

        This test checks if the create_dirs function creates directories corresponding to the predicted classes.
        """
        test_dir = os.path.join(self.outdir, "temp_dir")
        os.mkdir(test_dir)

        # Generate a DataFrame using class_prediction, which would be required by create_dirs
        df = class_prediction(self.model, self.classes, self.jpg_dir, self.outdir)
        
        create_dirs(df, test_dir)

        for model_class in self.classes:
            self.assertTrue(model_class in os.listdir(test_dir))
            os.rmdir(os.path.join(test_dir, model_class))
        
        os.rmdir(test_dir)

    def test_make_file_name(self):
        """
        Test the generation of a new file name.

        This test checks if the make_file_name function correctly generates a new file name.
        This test doesn't require model loading.
        """
        filename = "96957ff7-413f-4da2-b053-fdfa0bc3e290_label_front_0001_label_single.jpg"
        filename_stem = Path(filename).stem
        test_class = "handwritten"  # Use hardcoded class instead of self.classes[0]
        new_name = make_file_name(filename_stem, test_class)
        expected_name = f"96957ff7-413f-4da2-b053-fdfa0bc3e290_label_front_0001_label_single_{test_class}.jpg"
        self.assertEqual(new_name, expected_name)

    def test_filter_pictures(self):
        """
        Test the filtering of pictures based on class predictions.

        This test checks if the filter_pictures function filters pictures correctly based on class predictions.
        """
        # Generate a DataFrame using class_prediction, which would be required by filter_pictures
        df = class_prediction(self.model, self.classes, self.jpg_dir, self.outdir)

        # Call the filter_pictures function
        filter_pictures(self.jpg_dir, df, self.outdir)
        
        picture_count = 0
        for model_class in self.classes:
            picture_count += len(os.listdir(os.path.join(self.outdir, model_class)))

        # Check if the number of pictures in the directory matches
        self.assertEqual(len(os.listdir(self.jpg_dir)), picture_count)


class TestTFClassifierUtils(unittest.TestCase):
    """
    Test suite for utility functions that don't require model loading.
    """
    
    def test_make_file_name_simple(self):
        """Test make_file_name function with simple inputs."""
        result = make_file_name("test_label", "handwritten")
        expected = "test_label_handwritten.jpg"
        self.assertEqual(result, expected)
    
    def test_make_file_name_complex(self):
        """Test make_file_name function with complex label ID."""
        label_id = "96957ff7-413f-4da2-b053-fdfa0bc3e290_label_front_0001_label_single"
        result = make_file_name(label_id, "printed")
        expected = f"{label_id}_printed.jpg"
        self.assertEqual(result, expected)
    
    def test_create_dirs_simple(self):
        """Test create_dirs function with mock DataFrame."""
        import tempfile
        import shutil
        
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create a mock DataFrame
            df = pd.DataFrame({
                'filename': ['test1.jpg', 'test2.jpg'],
                'class': ['handwritten', 'printed']
            })
            
            # Test the function
            create_dirs(df, temp_dir)
            
            # Check if directories were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'handwritten')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'printed')))
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
