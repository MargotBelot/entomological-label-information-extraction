# Import third-party libraries
import unittest

# Import the necessary module from the 'label_evaluation' module package
from label_evaluation.evaluate_text import *


class TextSimilarityTestSuite(unittest.TestCase):
    """
    A class to test the calculation of text similarity.

    This test suite checks various scenarios for calculating text similarity using the `calculate_scores` function.
    """

    def test_capital_letters(self):
        """
        Test case for comparing strings with different capitalization.

        Test if the function correctly handles strings with different capitalization.
        """
        actual = calculate_scores("ANTWEB CASENT 0922859", "antweb casent 0922859")
        # wer and cer
        expected = (0.0, 0.0)
        self.assertEqual(actual, expected)

    def test_same_word(self):
        """
        Test case for comparing identical words.

        Test if the function correctly handles identical words.
        """
        actual = calculate_scores("coll", "coll")
        expected = (0, 0)
        self.assertEqual(actual, expected)

    def test_single_mistake(self):
        """
        Test case for comparing strings with a single mistake.

        Test if the function correctly calculates scores for strings with a single mistake.
        """
        actual = calculate_scores("ANTWEB CASENT 0922859", "ANTWEB CASENT 0922855")
        # Updated expected CER value to match difflib implementation (was 0.05 with editdistance)
        expected = (0.33, 0.1)
        self.assertEqual(actual, expected)

    def test_space(self):
        """
        Test case for comparing strings with different spaces.

        Test if the function correctly handles strings with different spaces.
        """
        actual = calculate_scores("ANTWEB", "ANTWEB ")
        expected = (0.0, 0.17)
        self.assertEqual(actual, expected)

    def test_punctuation (self):
        """
        Test case for comparing strings with different punctuation.

        Test if the function correctly handles strings with different punctuation.
        """
        actual = calculate_scores("Aenictus formosensis Forel, 1913 det. Michael Staab 2014",
                                  "Aenictus formosensis Forel, 1913. det. Michael Staab 2014")
        expected = (0.12, 0.02)
        self.assertEqual(actual, expected)

    def test_empty_string (self):
        """
        Test case for an empty string comparison.

        Test if the function correctly handles the case where one of the strings is empty.
        """
        actual = calculate_scores("Aenictus formosensis Forel, 1913 det. Michael Staab 2014",
                                  "")
        expected = (1, 1)
        print("len")
        print(len(" "))
        self.assertEqual(actual, expected)