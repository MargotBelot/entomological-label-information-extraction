# Import third-party libraries
import unittest
import json
import os

# Import the necessary module from the 'label_evaluation' module package
from label_postprocessing.ocr_postprocessing import *

class TestTextProcessing(unittest.TestCase):
    def setUp(self):
        """ Set up sample test data from OCR JSON file. """
        # Sample data for testing
        self.sample_data = [
            {"ID": "1", "text": "ECUADOR NAPO Prov"},
            {"ID": "2", "text": "http://example.com"},
            {"ID": "3", "text": " "},
            {"ID": "4", "text": "NAPO Prov."}
        ]
        self.test_output_file = "test_output.json"

        # Create a sample test file for OCR data
        with open(self.test_output_file, "w") as f:
            json.dump(self.sample_data, f)

    def tearDown(self):
        """ Clean up after tests. """
        if os.path.exists(self.test_output_file):
            os.remove(self.test_output_file)
        if os.path.exists("nuris.csv"):
            os.remove("nuris.csv")
        if os.path.exists("empty_transcripts.csv"):
            os.remove("empty_transcripts.csv")
        if os.path.exists("plausible_transcripts.json"):
            os.remove("plausible_transcripts.json")
        if os.path.exists("corrected_transcripts.json"):
            os.remove("corrected_transcripts.json")

    def test_count_mean_token_length(self):
        """ Test mean token length calculation. """
        tokens = ["ECUADOR", "NAPO", "Prov"]
        self.assertEqual(count_mean_token_length(tokens), 5.0)
        self.assertEqual(count_mean_token_length([]), 0)

    def test_is_plausible_prediction(self):
        """ Test plausible prediction detection. """
        self.assertTrue(is_plausible_prediction("ECUADOR NAPO Prov"))
        self.assertFalse(is_plausible_prediction("X!"))

    def test_correct_transcript(self):
        """ Test transcript correction. """
        self.assertEqual(correct_transcript("NAPO Prov."), "NAPO Prov")
        self.assertEqual(correct_transcript("00°39'10'S, 076° 26'W"), "003910S 076 26W")

    def test_is_nuri(self):
        """ Test NURI detection. """
        self.assertTrue(is_nuri("http://example.com"))
        self.assertFalse(is_nuri("NAPO Prov"))

    def test_is_empty(self):
        """ Test empty transcript detection. """
        self.assertTrue(is_empty(" "))
        self.assertFalse(is_empty("HOLOTYPE Camptocerus"))

    def test_save_transcripts(self):
        """ Test saving transcripts to CSV. """
        save_transcripts({"1": "ECUADOR"}, "test_transcripts.csv")
        self.assertTrue(os.path.exists("test_transcripts.csv"))
        os.remove("test_transcripts.csv")

    def test_save_json(self):
        """ Test saving transcripts to JSON. """
        save_json(self.sample_data, "test_output.json")
        self.assertTrue(os.path.exists("test_output.json"))
        os.remove("test_output.json")

    def test_process_ocr_output(self):
        """ Test OCR output processing with real dataset. """
        process_ocr_output(self.test_output_file)
        
        # Check that the output files were generated
        self.assertTrue(os.path.exists("nuris.csv"))
        self.assertTrue(os.path.exists("empty_transcripts.csv"))
        self.assertTrue(os.path.exists("plausible_transcripts.json"))
        self.assertTrue(os.path.exists("corrected_transcripts.json"))
        
        # Clean up the files
        os.remove("nuris.csv")
        os.remove("empty_transcripts.csv")
        os.remove("plausible_transcripts.json")
        os.remove("corrected_transcripts.json")

if __name__ == "__main__":
    unittest.main()
