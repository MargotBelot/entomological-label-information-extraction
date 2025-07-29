# Import third-party libraries
import unittest
import pandas as pd
from pathlib import Path

# Import the necessary module from the 'label_evaluation' module package
from label_evaluation.accuracy_classifier import *

class TestMetricsFunctions(unittest.TestCase):
    """Test suite for evaluating the performance of classification metrics functions."""

    def setUp(self):
        """Set up test data by reading the ground truth (gt) and predictions (pred) from a CSV file.
        
        This method loads the CSV file containing the ground truth and predicted labels, extracts the 
        unique class labels (target names), and prepares the output directory where results will be saved.
        """
        # Use absolute path based on test file location
        test_dir = Path(__file__).parent
        testdata_dir = test_dir.parent / "testdata"
        file_path = testdata_dir / "gt_pred_classiferHP.csv"
        
        if not file_path.exists():
            self.skipTest(f"Test data file not found: {file_path}")
            
        df = pd.read_csv(file_path, sep=';')
        
        self.pred = df['pred']
        self.gt = df['gt']
        self.target_names = list(set(self.gt.unique()))  # Extract unique class labels
        self.out_dir = testdata_dir / "output"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def test_metrics(self):
        """Test the output of the metrics function.
        
        This test verifies that the `metrics` function produces a non-empty report string, saves the 
        classification report to a file, and that the report contains an accuracy score.
        """
        report = metrics(self.target_names, self.pred, self.gt, self.out_dir)
        
        # Check if report is a non-empty string
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)
        
        # Check if the report file is created
        report_path = self.out_dir / "classification_report.txt"
        self.assertTrue(report_path.exists())
        
        # Check if the report contains accuracy score
        with open(report_path, 'r') as file:
            content = file.read()
            self.assertIn("Accuracy Score", content)

    def test_cm(self):
        """Test the confusion matrix function output.
        
        This test ensures that the `cm` function generates and saves a confusion matrix image
        in the specified output directory.
        """
        cm(self.target_names, self.pred, self.gt, self.out_dir)
        
        # Check if confusion matrix image is saved
        filename = f"{self.out_dir.stem}_cm.png"
        cm_path = self.out_dir / filename
        self.assertTrue(cm_path.exists())

    def tearDown(self):
        """Clean up the output directory by deleting all files after each test.
        
        This method removes all the files in the output directory to ensure a clean state for subsequent tests.
        Any file deletion errors (e.g., due to permissions) are handled with warnings.
        """
        for file in self.out_dir.glob("*"):
            try:
                file.unlink()
            except PermissionError:
                print(f"Warning: Could not delete {file}, skipping.")
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()