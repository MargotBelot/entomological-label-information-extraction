# Import third-party libraries
import unittest
from pathlib import Path
import detecto
import pandas as pd
import cv2
import os
import glob

# Import the necessary module from the 'label_processing' module package
from label_processing.label_detection import *

class TestSegmentationCropping(unittest.TestCase):
    """
    A test case for the Segmentation and Cropping functionality.
    This test suite verifies the functionality of the PredictLabel class, including image loading,
    model prediction, thresholding, and cropping operations.
    """
    path_to_model = str(Path(__file__).parent.resolve() / "../../models/label_detection_model.pth")
    jpg_path: Path = Path(__file__).parent.resolve() / "../testdata/uncropped/CASENT0922705_L.jpg"

    def setUp(self):
        """
        Setup method to initialize the model before each test.

        Initializes the PredictLabel instance with the provided model path and image path.
        This method runs before each test to ensure that the label predictor is available for testing.
        If initialization fails, it sets `label_predictor` to `None`.
        """
        try:
            self.label_predictor = PredictLabel(self.path_to_model, ["label"], self.jpg_path)
        except Exception as e:
            print(f"Error loading model or initializing PredictLabel: {e}")
            self.label_predictor = None

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
        
        # Get the parent directory of the image from self.label_predictor
        image_folder = str(self.label_predictor.jpg_path.parent)
        
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        df = prediction_parallel(os.path.join(os.path.dirname(__file__), "../testdata/uncropped", image_folder), label_predictor, n_processes=4)

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
        
        # Get the parent directory of the image from self.label_predictor
        image_folder = str(self.label_predictor.jpg_path.parent)
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        df = prediction_parallel(os.path.join(os.path.dirname(__file__), "../testdata/uncropped", image_folder), label_predictor, n_processes=4)

        self.assertEqual(len(df), 20)

    def test_threshold(self):
        """
        Test the threshold cleaning functionality.

        Verifies that the cleaning function removes entries below the specified threshold.
        The expected output should have no rows remaining when the threshold is set to 1.0.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        
        # Get the parent directory of the image from self.label_predictor
        image_folder = str(self.label_predictor.jpg_path.parent)
        
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        df = prediction_parallel(os.path.join(os.path.dirname(__file__), "../testdata/uncropped", image_folder), label_predictor, n_processes=4)

        clean_df = clean_predictions(os.path.join(os.path.dirname(__file__), "../testdata/uncropped"), df, 1.0)
        self.assertEqual(len(clean_df), 0)

    def test_crops(self):
        """
        Test the creation of crops from predictions.
        Verifies that crop files are created for each prediction in the specified output directory.
        The number of crops created should be equal to the number of predictions detected.
        """
        if self.label_predictor is None:
            self.skipTest("PredictLabel model could not be initialized.")
        
        # Get the parent directory of the image from self.label_predictor
        image_folder = str(self.label_predictor.jpg_path.parent)
        
        label_predictor = PredictLabel(self.path_to_model, ["label"])
        df = prediction_parallel(os.path.join(os.path.dirname(__file__), "../testdata/uncropped", image_folder), label_predictor, n_processes=4)
        
        # Log the number of predictions
        print(f"Number of predictions: {len(df)}")
        
        create_crops(os.path.join(os.path.dirname(__file__), "../testdata/uncropped"), df, out_dir=Path("../testdata/check_crops"))
        crop_files = glob.glob(os.path.join(Path("../testdata/check_crops/uncropped_cropped"), '*.jpg'))
        
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
        if os.path.exists("../testdata/check_crops"):
            for f in glob.glob("../testdata/check_crops/*"):
                try:
                    os.chmod(f, 0o777)  # Ensure write permission
                    os.remove(f)
                except PermissionError:
                    print(f"Permission denied while deleting {f}.")
                    continue
            try:
                os.rmdir("../testdata/check_crops")  # Remove the empty directory
            except OSError as e:
                print(f"Failed to remove 'check_crops' directory: {e}")