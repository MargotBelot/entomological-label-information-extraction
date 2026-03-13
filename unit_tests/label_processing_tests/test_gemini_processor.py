"""Tests for label_processing.gemini_processor — pure-logic functions only (no API calls)."""

import unittest
import numpy as np

from label_processing.gemini_processor import _rescale_bbox, rotate_image


class TestRescaleBbox(unittest.TestCase):
    """Tests for the _rescale_bbox coordinate conversion function."""

    def test_full_image_bbox(self):
        """A bbox spanning the entire 0-1000 range maps to the full image dimensions."""
        bbox = {"top": 0, "left": 0, "bottom": 1000, "right": 1000}
        result = _rescale_bbox(bbox, img_w=2000, img_h=1000, padding_pct=0.0)
        self.assertEqual(result, {"xmin": 0, "ymin": 0, "xmax": 2000, "ymax": 1000})

    def test_partial_bbox_no_padding(self):
        """A centered quarter-box converts correctly without padding."""
        bbox = {"top": 250, "left": 250, "bottom": 750, "right": 750}
        result = _rescale_bbox(bbox, img_w=1000, img_h=1000, padding_pct=0.0)
        self.assertEqual(result, {"xmin": 250, "ymin": 250, "xmax": 750, "ymax": 750})

    def test_default_padding(self):
        """Default 2% padding expands the crop by the expected pixel amount."""
        bbox = {"top": 500, "left": 500, "bottom": 600, "right": 600}
        result = _rescale_bbox(bbox, img_w=1000, img_h=1000)  # default 2%
        self.assertEqual(result["xmin"], 500 - 20)
        self.assertEqual(result["ymin"], 500 - 20)
        self.assertEqual(result["xmax"], 600 + 20)
        self.assertEqual(result["ymax"], 600 + 20)

    def test_padding_clamps_to_zero(self):
        """Padding never produces negative coordinates."""
        bbox = {"top": 0, "left": 0, "bottom": 100, "right": 100}
        result = _rescale_bbox(bbox, img_w=1000, img_h=1000, padding_pct=0.05)
        self.assertEqual(result["xmin"], 0)
        self.assertEqual(result["ymin"], 0)

    def test_padding_clamps_to_image_size(self):
        """Padding never exceeds image dimensions."""
        bbox = {"top": 900, "left": 900, "bottom": 1000, "right": 1000}
        result = _rescale_bbox(bbox, img_w=500, img_h=500, padding_pct=0.05)
        self.assertLessEqual(result["xmax"], 500)
        self.assertLessEqual(result["ymax"], 500)

    def test_missing_keys_use_defaults(self):
        """Missing top/left default to 0, bottom/right default to 1000."""
        bbox = {}
        result = _rescale_bbox(bbox, img_w=800, img_h=600, padding_pct=0.0)
        self.assertEqual(result, {"xmin": 0, "ymin": 0, "xmax": 800, "ymax": 600})

    def test_non_square_image(self):
        """Coordinates scale independently for width vs height."""
        bbox = {"top": 500, "left": 500, "bottom": 500, "right": 500}
        result = _rescale_bbox(bbox, img_w=2000, img_h=1000, padding_pct=0.0)
        self.assertEqual(result["xmin"], 1000)  # 500/1000 * 2000
        self.assertEqual(result["ymin"], 500)   # 500/1000 * 1000


class TestRotateImage(unittest.TestCase):
    """Tests for the rotate_image helper."""

    def _make_image(self, h=100, w=150, channels=3):
        return np.zeros((h, w, channels), dtype=np.uint8)

    def test_zero_angle_returns_same(self):
        """An angle < 0.5° returns the original image unchanged."""
        img = self._make_image()
        rotated = rotate_image(img, 0.0)
        self.assertTrue(np.array_equal(rotated, img))

    def test_small_angle_returns_same(self):
        """An angle of 0.3° (< 0.5 threshold) returns original."""
        img = self._make_image()
        rotated = rotate_image(img, 0.3)
        self.assertTrue(np.array_equal(rotated, img))

    def test_90_degree_rotation_dimensions(self):
        """A 90° rotation swaps width and height (approximately)."""
        img = self._make_image(h=100, w=200)
        rotated = rotate_image(img, 90.0)
        # After 90° rotation, new width ≈ old height, new height ≈ old width
        self.assertAlmostEqual(rotated.shape[0], 200, delta=2)
        self.assertAlmostEqual(rotated.shape[1], 100, delta=2)

    def test_180_degree_rotation_preserves_size(self):
        """A 180° rotation keeps the same dimensions."""
        img = self._make_image(h=100, w=200)
        rotated = rotate_image(img, 180.0)
        self.assertEqual(rotated.shape[:2], img.shape[:2])

    def test_output_is_numpy_array(self):
        """Output should always be a numpy ndarray."""
        img = self._make_image()
        rotated = rotate_image(img, 45.0)
        self.assertIsInstance(rotated, np.ndarray)

    def test_output_has_three_channels(self):
        """A 3-channel input produces a 3-channel output."""
        img = self._make_image()
        rotated = rotate_image(img, 30.0)
        self.assertEqual(len(rotated.shape), 3)
        self.assertEqual(rotated.shape[2], 3)

    def test_arbitrary_angle_expands_canvas(self):
        """Non-orthogonal rotation produces a larger canvas than the original."""
        img = self._make_image(h=100, w=100)
        rotated = rotate_image(img, 45.0)
        self.assertGreater(rotated.shape[0], 100)
        self.assertGreater(rotated.shape[1], 100)


if __name__ == "__main__":
    unittest.main()
