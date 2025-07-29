#!/usr/bin/env python3
"""
Basic CI verification test - should pass in CI environment.
This tests the most essential functionality without complex dependencies.
"""

import unittest
import os
import sys
from pathlib import Path

class BasicCITest(unittest.TestCase):
    """Basic tests that should always pass in CI."""
    
    def test_python_basics(self):
        """Test basic Python functionality."""
        self.assertTrue(True)
        self.assertEqual(2 + 2, 4)
        self.assertIsInstance("hello", str)
    
    def test_pathlib_works(self):
        """Test that pathlib works correctly."""
        current_dir = Path(__file__).parent
        self.assertTrue(current_dir.exists())
        self.assertTrue(current_dir.is_dir())
    
    def test_environment_variables(self):
        """Test that required environment variables are set."""
        # These should be set by CI, but may not be set locally
        ci_env_vars = ['TF_CPP_MIN_LOG_LEVEL', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH']
        
        # Check if we're in CI (GitHub Actions sets CI=true)
        is_ci = os.environ.get('CI', '').lower() == 'true'
        
        if is_ci:
            for var in ci_env_vars:
                self.assertIn(var, os.environ, f"CI environment variable {var} not set")
        else:
            # Just check that we can set them if needed
            for var in ci_env_vars:
                if var not in os.environ:
                    os.environ[var] = 'test_value'
    
    def test_testdata_structure(self):
        """Test that basic test data structure exists."""
        testdata_dir = Path("unit_tests/testdata")
        self.assertTrue(testdata_dir.exists(), f"Test data directory missing: {testdata_dir}")
        
        # Essential files
        essential_files = [
            testdata_dir / "iou_scores.csv",
            testdata_dir / "gt_pred_classiferHP.csv"
        ]
        
        for file_path in essential_files:
            self.assertTrue(file_path.exists(), f"Essential test file missing: {file_path}")
    
    def test_output_directory_creation(self):
        """Test that output directory can be created."""
        output_dir = Path("unit_tests/testdata/output")
        output_dir.mkdir(exist_ok=True)
        self.assertTrue(output_dir.exists())
        
        # Test write permissions
        test_file = output_dir / "ci_test.tmp"
        test_file.write_text("CI test")
        self.assertTrue(test_file.exists())
        test_file.unlink()  # Clean up
    
    def test_imports_basic(self):
        """Test basic imports that should work in CI."""
        try:
            import pandas as pd
            import numpy as np
            from pathlib import Path
            import os
            import unittest
        except ImportError as e:
            self.fail(f"Basic import failed: {e}")
    
    def test_imports_advanced(self):
        """Test advanced imports that might fail in CI."""
        try:
            import tensorflow as tf
            # Just test import, not full functionality
            self.assertIsNotNone(tf.__version__)
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
        except Exception as e:
            self.fail(f"TensorFlow import failed: {e}")

if __name__ == '__main__':
    # Run with maximum verbosity
    unittest.main(verbosity=2)
