# Import third-party libraries
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Import the necessary module from the 'label_evaluation' module package
from label_evaluation.iou_scores import *

class TestIOUModule(unittest.TestCase):
    """Test suite for evaluating the IOU (Intersection Over Union) functions in the label_evaluation module."""

    def setUp(self):
        """Set up test data and output directory for IOU tests.
        
        This method loads a CSV file containing the necessary test data and prepares the output directory
        where results will be saved or plotted.
        """
        file_path = "../testdata/iou_scores.csv"
        self.df_concat = pd.read_csv(file_path)
        
        self.out_dir = Path("../testdata/output")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def test_calculate_iou(self):
        """Test the calculate_iou function.
        
        This test checks if the calculate_iou function computes the IOU score for bounding boxes 
        correctly. It ensures that the returned score is within the valid range [0, 1].
        """
        iou = calculate_iou((259, 225, 2309, 860), ("label", 258.76, 222.87, 2310.09, 874.00))
        self.assertGreaterEqual(iou, 0)  # IOU score should be greater than or equal to 0
        self.assertLessEqual(iou, 1)  # IOU score should be less than or equal to 1

    def test_comparison(self):
        """Test the comparison function using real data.
        
        This test evaluates the comparison function, which compares predicted bounding boxes to ground truth 
        boxes and calculates IOU scores. It ensures that the output DataFrame is not empty and contains 
        a column for the score.
        """
        result_df = comparison(self.df_concat[['filename', 'class_pred', 'xmin_pred', 'ymin_pred', 'xmax_pred', 'ymax_pred']], 
                               self.df_concat[['filename', 'class_gt', 'xmin_gt', 'ymin_gt', 'xmax_gt', 'ymax_gt']])
        self.assertFalse(result_df.empty)  # Ensure that the result DataFrame is not empty
        self.assertIn("score", result_df.columns)  # Check if the score column is present

    def test_concat_frames(self):
        """Test the concat_frames function using real data.
        
        This test checks the functionality of the concat_frames function, which concatenates predicted and 
        ground truth bounding boxes, and calculates the IOU score for each pair. The test ensures that 
        the concatenated DataFrame contains the 'score' column and is not empty.
        """
        concatenated_df = concat_frames(self.df_concat[['filename', 'class_pred', 'xmin_pred', 'ymin_pred', 'xmax_pred', 'ymax_pred']], 
                                        self.df_concat[['filename', 'class_gt', 'xmin_gt', 'ymin_gt', 'xmax_gt', 'ymax_gt']])
        self.assertFalse(concatenated_df.empty)  # Ensure the concatenated DataFrame is not empty
        self.assertIn("score", concatenated_df.columns)  # Check for 'score' column in the output

    def test_box_plot_iou(self):
        """Test the box plot generation function.
        
        This test verifies that the box_plot_iou function generates a box plot correctly and saves the 
        plot as a figure. It ensures that the generated figure is not None.
        """
        fig = box_plot_iou(self.df_concat, self.out_dir / "accuracy.txt")
        self.assertIsNotNone(fig)  # Ensure that the figure is not None

    def tearDown(self):
        """Clean up test output files and directories.
        
        This method removes all files generated during the tests from the output directory. It handles 
        permission errors and attempts to remove the directory if it is empty.
        """
        for file in self.out_dir.iterdir():
            try:
                file.unlink()  # Remove file
            except PermissionError:
                print(f"Warning: Could not delete {file}, skipping.")
            except FileNotFoundError:
                pass  # File might have been deleted already
        try:
            self.out_dir.rmdir()  # Remove the directory if it's empty
        except OSError:
            print(f"Warning: Could not remove directory {self.out_dir}, skipping.")

if __name__ == "__main__":
    unittest.main()
