# Import third-party libraries
import unittest
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import the necessary module from the 'label_processing' module package
from label_processing.utils import *

class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions in the 'label_processing' module."""

    # Test check_dir function
    @patch('os.path.isdir')
    @patch('os.listdir')
    def test_check_dir(self, mock_listdir, mock_isdir):
        """
        Test the check_dir function, which checks if a directory contains .jpg files.

        This test verifies the following:
        The function correctly handles valid directories containing .jpg files.
        The function raises a FileNotFoundError when no .jpg files are present.
        The function raises a FileNotFoundError for invalid directories.
        """
        # Test with valid directory and jpg files
        mock_isdir.return_value = True
        mock_listdir.return_value = ['image1.jpg', 'image2.jpg']
        
        try:
            check_dir('/valid/dir')  # Mocked directory path
        except FileNotFoundError:
            self.fail("check_dir raised FileNotFoundError unexpectedly!")
        
        # Test with no jpg files
        mock_listdir.return_value = ['image1.png', 'image2.png']
        with self.assertRaises(FileNotFoundError):
            check_dir('/valid/dir')  # Mocked directory path
        
        # Test with invalid directory
        mock_isdir.return_value = False
        with self.assertRaises(FileNotFoundError):
            check_dir('/invalid/dir')  # Mocked directory path

    # Test generate_filename function
    def test_generate_filename(self):
        """
        Test the generate_filename function that creates filenames based on input paths and an appendix.

        This test checks the following cases:
        If the function generates the correct filename by appending a string to the original path.
        If the function works when a directory is provided instead of a file.
        If the function adds the appropriate file extension when provided.
        """
        # Test with a valid file path
        original_path = Path(__file__).parent / '..' / 'testdata' / 'cropped_pictures' / 'BLF1562(11)-2_L_label_typed_3_printed.jpg'  # Define the file path
        appendix = 'processed'
        expected_filename = 'BLF1562(11)-2_L_label_typed_3_printed_processed'  # Expected output file name
        result = generate_filename(original_path, appendix)
        self.assertEqual(result, expected_filename)
        
        # Test with directory path (no file extension)
        original_path = Path(__file__).parent / '..' / 'testdata'  # Define directory path
        expected_filename = 'testdata_processed'  # Expected output directory name
        result = generate_filename(original_path, appendix)
        self.assertEqual(result, expected_filename)
        
        # Test with extension passed
        original_path = Path(__file__).parent / '..' / 'testdata' / 'cropped_pictures' / 'BLF1562(11)-2_L_label_typed_3_printed.jpg'
        expected_filename = 'BLF1562(11)-2_L_label_typed_3_printed_processed.jpg'
        result = generate_filename(original_path, appendix, extension='jpg')
        self.assertEqual(result, expected_filename)

    # Test save_json function
    def test_save_json(self):
        """
        Test the save_json function to save a dictionary to a JSON file.

        This test checks:
        If the function correctly saves data to a JSON file in the specified directory.
        If the saved file exists and contains the expected data.
        """
        # Create temporary directory to save the file
        with tempfile.TemporaryDirectory() as temp_dir:
            data = [{'ID': '123', 'text': 'example'}]
            filename = 'test_output.json'
            
            # Ensure directory exists before saving
            temp_data_dir = os.path.join(temp_dir, "testdata")  # Define where the file will be saved
            os.makedirs(temp_data_dir, exist_ok=True)

            save_json(data, filename, temp_data_dir)
            
            # Check if the file is created
            filepath = os.path.join(temp_data_dir, filename)  # Final path to check
            self.assertTrue(os.path.exists(filepath))
            
            # Check file content
            with open(filepath, 'r', encoding='utf8') as f:
                loaded_data = json.load(f)
                self.assertEqual(loaded_data, data)

    # Test replace_nuri function
    def test_replace_nuri(self):
        """
        Test the replace_nuri function, which replaces NURI patterns with URLs.

        This test checks:
        If the function correctly replaces NURI patterns like '_u_123abc' with the corresponding URL.
        If the function handles Picturae NURI patterns correctly (e.g., '_u_abcdef123456.jpg').
        If no NURI pattern results in no change to the text.
        """
        # Test valid NURI format
        transcript = {'ID': '_u_123abc', 'text': 'Some text with _u_123abc'}
        result = replace_nuri(transcript)
        self.assertEqual(result['text'], 'http://coll.mfn-berlin.de/u/123abc')
        
        # Test valid Picturae NURI format
        transcript = {'ID': '_u_abcdef123456.jpg', 'text': 'Text with _u_abcdef123456.jpg'}
        result = replace_nuri(transcript)
        self.assertEqual(result['text'], 'http://coll.mfn-berlin.de/u/abcdef123456')
        
        # Test when no NURI pattern is present
        transcript = {'ID': 'no_nuri_pattern', 'text': 'No NURI pattern here'}
        result = replace_nuri(transcript)
        self.assertEqual(result['text'], 'No NURI pattern here')

if __name__ == '__main__':
    unittest.main()
