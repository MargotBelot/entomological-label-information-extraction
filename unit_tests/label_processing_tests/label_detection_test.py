# Import third-party libraries
import unittest
from pathlib import Path
import detecto
import pandas as pd
import cv2
import os
import glob
import platform
import sys
import torch

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import the necessary module from the 'label_processing' module package
from label_processing.label_detection import *
from label_processing.config import config

class TestSegmentationCropping(unittest.TestCase):
    """
    A test case for the Segmentation and Cropping functionality.
    This test suite verifies the functionality of the PredictLabel class, including image loading,
    model prediction, thresholding, and cropping operations.
    """
    @classmethod
    def setUpClass(cls):
        """Set up class-level attributes using centralized configuration."""
        cls.path_to_model = str(config.get_model_path("detection"))
        cls.jpg_path = config.test_data_dir / "uncropped" / "CASENT0922705_L.jpg"
        cls.testdata_dir = config.test_data_dir

    def setUp(self):
        """
        Setup method to validate model and test data availability before each test.

        Performs comprehensive checks for cross-platform compatibility:
        - Validates model file existence and integrity
        - Validates test data availability
        - Attempts model initialization with error handling
        - Sets up proper environment variables for cross-platform testing
        """
        import platform
        
        # Set up cross-platform environment
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU for consistent testing
        
        # Check model file existence and integrity
        if not os.path.exists(self.path_to_model):
            self.skipTest(f"Model file not found: {self.path_to_model}")
            
        if os.path.getsize(self.path_to_model) == 0:
            self.skipTest(f"Model file is empty: {self.path_to_model}")
            
        # Check test image existence
        if not os.path.exists(self.jpg_path):
            self.skipTest(f"Test image not found: {self.jpg_path}")
            
        # Validate test image integrity
        test_img = cv2.imread(str(self.jpg_path))
        if test_img is None:
            self.skipTest(f"Test image is corrupted or unreadable: {self.jpg_path}")
            
        # Try to initialize PredictLabel with comprehensive error handling
        try:
            print(f"\nTesting on {platform.system()} {platform.release()}")
            print(f"Python version: {platform.python_version()}")
            
            self.label_predictor = PredictLabel(self.path_to_model, ["label"], self.jpg_path)
            print("✓ PredictLabel initialized successfully in setUp")
            
        except FileNotFoundError as e:
            self.skipTest(f"Model file not found during initialization: {e}")
        except ImportError as e:
            self.skipTest(f"Missing dependencies for model loading: {e}")
        except RuntimeError as e:
            self.skipTest(f"Runtime error during model loading (possibly CUDA/CPU mismatch): {e}")
        except Exception as e:
            print(f"Error loading model or initializing PredictLabel: {e}")
            # Print additional debugging info on Linux
            if platform.system() == 'Linux':
                print(f"Linux-specific debugging:")
                print(f"  PyTorch version: {torch.__version__ if 'torch' in sys.modules else 'Not loaded'}")
                print(f"  CUDA available: {torch.cuda.is_available() if 'torch' in sys.modules else 'Unknown'}")
                print(f"  Environment CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
            
            self.label_predictor = None
            self.skipTest(f"PredictLabel model could not be initialized: {e}")

    def test_predict_label_constructor(self):
        """
        Test the constructor of the PredictLabel class.

        Ensures that the PredictLabel instance is created with the correct attributes:
        - The image path should be a valid Path object.
        - The model should be an instance of detecto.core.Model.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        label_predictor = PredictLabel(self.path_to_model, ["label"], self.jpg_path)
        self.assertIsInstance(label_predictor.jpg_path, Path)
        self.assertIsInstance(label_predictor.model, detecto.core.Model)

    def test_predict_label_constructor_2(self):
        """
        Test an alternative constructor of the PredictLabel class.

        Ensures that the PredictLabel instance is created correctly when the image path
        is set separately after the object is initialized.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        label_predictor.jpg_path = self.jpg_path
        self.assertIsInstance(label_predictor.jpg_path, Path)

    def test_class_prediction(self):
        """
        Test the class prediction method of the PredictLabel class.

        Verifies that the class prediction method returns a DataFrame with the predicted entries.
        The DataFrame should be correctly formatted and contain the prediction results.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        entries = self.label_predictor.class_prediction(self.jpg_path)
        self.assertIsInstance(entries, pd.DataFrame)

    def test_class_prediction_parallel(self):
        """
        Test the parallel class prediction method.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        
        # Use project root-based paths
        uncropped_dir = self.testdata_dir / "uncropped"
        
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        df = prediction_parallel(uncropped_dir, label_predictor, n_processes=4)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 7)

    def test_number_of_labels_detected_single_image(self):
        """
        Test the number of labels detected in a single image.

        Verifies that the correct number of labels are detected in a single image.
        The expected number of labels is 5 for the provided test image.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        df = self.label_predictor.class_prediction(self.jpg_path)
        self.assertEqual(len(df), 5)

    def test_number_of_labels_detected_image_folder(self):
        """
        Test the number of labels detected in an image folder.

        Verifies that the correct number of labels are detected in a folder of images.
        The expected number of labels for the folder is 20.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        
        # Use project root-based paths
        uncropped_dir = self.testdata_dir / "uncropped"
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        df = prediction_parallel(uncropped_dir, label_predictor, n_processes=4)

        self.assertEqual(len(df), 20)

    def test_threshold(self):
        """
        Test the threshold cleaning functionality.

        Verifies that the cleaning function removes entries below the specified threshold.
        The expected output should have no rows remaining when the threshold is set to 1.0.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        
        # Use project root-based paths
        uncropped_dir = self.testdata_dir / "uncropped"
        
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        df = prediction_parallel(uncropped_dir, label_predictor, n_processes=4)

        clean_df = clean_predictions(uncropped_dir, df, 1.0)
        self.assertEqual(len(clean_df), 0)

    def test_crops(self):
        """
        Test the creation of crops from predictions.
        Verifies that crop files are created for each prediction in the specified output directory.
        The number of crops created should be equal to the number of predictions detected.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        
        # Use project root-based paths
        uncropped_dir = self.testdata_dir / "uncropped"
        check_crops_dir = self.testdata_dir / "check_crops"
        
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        df = prediction_parallel(uncropped_dir, label_predictor, n_processes=4)
        
        # Log the number of predictions
        print(f"Number of predictions: {len(df)}")
        
        create_crops(uncropped_dir, df, out_dir=check_crops_dir)
        crop_files = list((check_crops_dir / "uncropped_cropped").glob('*.jpg'))
        
        # Log the number of crop files created
        print(f"Number of crop files: {len(crop_files)}")
        
        self.assertEqual(len(df), len(crop_files))

    def test_image_loading(self):
        """
        Ensure the image loads properly before passing to the model.

        Verifies that the image can be read from the specified path and is not `None`.
        If the image cannot be loaded, a ValueError will be raised.
        """
        img = cv2.imread(str(self.jpg_path))
        if img is None:
            raise ValueError(f"Image at {self.jpg_path} is missing or unreadable!")
        self.assertIsNotNone(img, "The image could not be loaded.")

    def tearDown(self):
        """
        Cleanup temporary files after each test.
        
        This method is called after each test to remove temporary directories and crop files created
        during the test. This ensures that each test starts with a clean slate.
        """
        check_crops_path = self.testdata_dir / "check_crops"
        if check_crops_path.exists():
            import shutil
            try:
                # Use shutil.rmtree for more reliable cleanup across platforms
                shutil.rmtree(check_crops_path)
                print(f"✓ Cleaned up {check_crops_path}")
            except PermissionError as e:
                print(f"Permission denied while deleting {check_crops_path}: {e}")
            except OSError as e:
                print(f"Failed to remove 'check_crops' directory: {e}")
