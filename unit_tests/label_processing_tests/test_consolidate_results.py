# Import third-party libraries
import unittest
import os
import csv
import tempfile
from pathlib import Path

# Import the consolidate_results module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'postprocessing'))
from consolidate_results import load_rotation_results


class TestConsolidateResults(unittest.TestCase):
    """
    A test case for the consolidate_results script.
    
    This test suite ensures that rotation metadata loading works correctly
    from multiple possible directory locations.
    """
    
    def test_rotation_metadata_from_main_directory(self):
        """
        Test that rotation metadata can be loaded from the main output directory.
        
        This tests the first path checked in load_rotation_results (line 146):
        os.path.join(output_dir, 'rotation_metadata.csv')
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create rotation_metadata.csv in main directory
            meta_file = os.path.join(temp_dir, 'rotation_metadata.csv')
            
            with open(meta_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'angle', 'corrected'])
                writer.writerow(['image1.jpg', '90', 'True'])
                writer.writerow(['image2.jpg', '0', 'False'])
            
            # Load rotation results
            results = load_rotation_results(temp_dir)
            
            # Verify results
            self.assertEqual(len(results), 2, "Should load 2 rotation entries")
            self.assertIn('image1.jpg', results)
            self.assertEqual(results['image1.jpg']['angle'], 90)
            self.assertTrue(results['image1.jpg']['corrected'])
            
            self.assertIn('image2.jpg', results)
            self.assertEqual(results['image2.jpg']['angle'], 0)
            self.assertFalse(results['image2.jpg']['corrected'])
    
    def test_rotation_metadata_from_printed_preprocessed_directory(self):
        """
        Test that rotation metadata can be loaded from the printed_preprocessed subdirectory.
        
        This tests the second path checked in load_rotation_results (line 147):
        os.path.join(output_dir, 'printed_preprocessed', 'rotation_metadata.csv')
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create printed_preprocessed subdirectory
            subdir = os.path.join(temp_dir, 'printed_preprocessed')
            os.makedirs(subdir, exist_ok=True)
            
            # Create rotation_metadata.csv in subdirectory
            meta_file = os.path.join(subdir, 'rotation_metadata.csv')
            
            with open(meta_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'angle', 'corrected'])
                writer.writerow(['image3.jpg', '180', 'True'])
            
            # Load rotation results
            results = load_rotation_results(temp_dir)
            
            # Verify results
            self.assertEqual(len(results), 1, "Should load 1 rotation entry")
            self.assertIn('image3.jpg', results)
            self.assertEqual(results['image3.jpg']['angle'], 180)
            self.assertTrue(results['image3.jpg']['corrected'])
    
    def test_rotation_metadata_from_printed_rotated_directory(self):
        """
        Test that rotation metadata can be loaded from the printed_rotated subdirectory.
        
        This tests the third path checked in load_rotation_results (line 148):
        os.path.join(output_dir, 'printed_rotated', 'rotation_metadata.csv')
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create printed_rotated subdirectory
            subdir = os.path.join(temp_dir, 'printed_rotated')
            os.makedirs(subdir, exist_ok=True)
            
            # Create rotation_metadata.csv in subdirectory
            meta_file = os.path.join(subdir, 'rotation_metadata.csv')
            
            with open(meta_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'angle', 'corrected'])
                writer.writerow(['image4.jpg', '270', 'True'])
            
            # Load rotation results
            results = load_rotation_results(temp_dir)
            
            # Verify results
            self.assertEqual(len(results), 1, "Should load 1 rotation entry")
            self.assertIn('image4.jpg', results)
            self.assertEqual(results['image4.jpg']['angle'], 270)
            self.assertTrue(results['image4.jpg']['corrected'])
    
    def test_rotation_metadata_priority_order(self):
        """
        Test that rotation metadata is loaded from the first available location.
        
        When metadata exists in multiple locations, the function should load from
        the first one it finds (main dir > printed_preprocessed > printed_rotated).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create metadata in main directory (should be loaded)
            main_meta_file = os.path.join(temp_dir, 'rotation_metadata.csv')
            with open(main_meta_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'angle', 'corrected'])
                writer.writerow(['main_image.jpg', '90', 'True'])
            
            # Create metadata in subdirectory (should be ignored)
            subdir = os.path.join(temp_dir, 'printed_preprocessed')
            os.makedirs(subdir, exist_ok=True)
            sub_meta_file = os.path.join(subdir, 'rotation_metadata.csv')
            with open(sub_meta_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'angle', 'corrected'])
                writer.writerow(['sub_image.jpg', '180', 'True'])
            
            # Load rotation results
            results = load_rotation_results(temp_dir)
            
            # Verify only main directory metadata was loaded
            self.assertEqual(len(results), 1, "Should load only from first location")
            self.assertIn('main_image.jpg', results)
            self.assertNotIn('sub_image.jpg', results)
    
    def test_rotation_metadata_no_file_found(self):
        """
        Test that when no rotation metadata file exists, an empty dict is returned.
        
        This tests the fallback behavior when none of the expected paths exist.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Don't create any metadata files
            
            # Load rotation results
            results = load_rotation_results(temp_dir)
            
            # Verify empty dict is returned
            self.assertIsInstance(results, dict, "Should return a dict")
            self.assertEqual(len(results), 0, "Should return empty dict when no metadata found")


if __name__ == '__main__':
    unittest.main()
