#!/usr/bin/env python3
"""
Gemini Processor Module

Uses the Google GenAI SDK (google-genai) to:
1. Classify detected labels (empty, identifier, printed, handwritten, mixed)
2. Determine rotation angle for each label
3. Perform OCR/HTR on label images (printed, handwritten, and mixed)

All operations use the Gemini vision model to process images.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types


# ---------------------Gemini Client---------------------#


def get_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Create a Gemini API client.

    Args:
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.

    Returns:
        genai.Client: Authenticated Gemini client.
    """
    if api_key:
        return genai.Client(api_key=api_key)

    # Fall back to environment variable (auto-detected by SDK)
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not env_key:
        raise ValueError(
            "No Gemini API key provided. Set GEMINI_API_KEY environment variable "
            "or pass api_key argument."
        )
    return genai.Client(api_key=env_key)


def _encode_image(image_path: str) -> tuple[bytes, str]:
    """
    Read an image file and return its bytes and MIME type.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (image_bytes, mime_type).
    """
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    return image_bytes, mime_type


# ---------------------Classification + Rotation---------------------#


# Prompt for classifying labels and detecting rotation in a full specimen image.
# Gemini receives the image + bounding box coordinates from detection.
CLASSIFICATION_PROMPT = """You are analyzing an entomological specimen image. 
Labels have been detected at the following bounding box coordinates (xmin, ymin, xmax, ymax):

{bounding_boxes}

For EACH detected label, analyze the region and return a JSON array with one object per label:

{{
  "labels": [
    {{
      "label_index": <int>,
      "xmin": <int>,
      "ymin": <int>,
      "xmax": <int>,
      "ymax": <int>,
      "category": "<empty|identifier|printed|handwritten|mixed>",
      "rotation_angle": <float>,
      "confidence": <float 0-1>
    }}
  ]
}}

Category definitions:
- "empty": blank label with no visible text or content
- "identifier": label whose PRIMARY content is a QR code, barcode, data-matrix, or
  machine-readable identifier. A label is still "identifier" even if it contains a small
  amount of supporting printed text such as a URI, catalogue number, or institution code
  alongside the code/barcode. Only classify as "mixed" if the label has substantial
  handwritten AND printed text — NOT just because it has a code plus a short text line.
- "printed": label with only typed/printed text (no barcodes/QR codes as primary element)
- "handwritten": label with only handwritten text
- "mixed": label containing BOTH printed AND handwritten text in significant amounts

rotation_angle: degrees to rotate CLOCKWISE to make the text horizontal and readable.
Use 0 if already correctly oriented. Can be any angle (not just 0/90/180/270).

Return ONLY valid JSON, no extra text."""


def classify_and_detect_rotation(
    image_path: str,
    bounding_boxes: list[dict],
    client: genai.Client,
    model: str = "gemini-2.5-flash",
) -> list[dict]:
    """
    Classify detected labels and determine rotation angles in a single API call.

    Sends the full specimen image with bounding box coordinates to Gemini.
    Returns per-label classification and rotation angle.

    Args:
        image_path: Path to the full specimen image.
        bounding_boxes: List of dicts with keys: label_index, xmin, ymin, xmax, ymax.
        client: Authenticated Gemini client.
        model: Gemini model to use.

    Returns:
        List of dicts with: label_index, xmin, ymin, xmax, ymax,
        category, rotation_angle, confidence.
    """
    if not bounding_boxes:
        return []

    image_bytes, mime_type = _encode_image(image_path)

    # Format bounding box info for the prompt
    bbox_text = "\n".join(
        f"  Label {bb['label_index']}: ({bb['xmin']}, {bb['ymin']}, {bb['xmax']}, {bb['ymax']})"
        for bb in bounding_boxes
    )
    prompt = CLASSIFICATION_PROMPT.format(bounding_boxes=bbox_text)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    # Parse the JSON response
    try:
        result = json.loads(response.text)
        labels = result.get("labels", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Warning: Failed to parse Gemini classification response: {e}")
        print(f"Raw response: {response.text[:500]}")
        labels = []

    return labels


# ---------------------Gemini Detection + Classification---------------------#


DETECTION_CLASSIFICATION_PROMPT = """You are analyzing an image that contains one or more labels.
Labels can be any piece of paper, card, sticker, tag, or printed/handwritten note visible
in the image. They may be attached to a specimen, pinned, glued, or simply present.

Your task:
1. DETECT every distinct label visible in the image — regardless of size, shape, or content.
   This includes printed labels, handwritten notes, barcodes, QR codes, colour references,
   stickers, blank cards, and any other discrete piece of paper or tag.
2. For each label, provide a bounding box AND classify it.

COORDINATE SYSTEM (important):
- All coordinates use a 0–1000 normalised scale.
- "top"    = distance from the TOP    edge of the image (0 = very top,    1000 = very bottom).
- "left"   = distance from the LEFT   edge of the image (0 = very left,   1000 = very right).
- "bottom" = distance from the TOP    edge to the BOTTOM of the box.
- "right"  = distance from the LEFT   edge to the RIGHT  side of the box.
- So top < bottom, and left < right.

Return a JSON object with the following structure:

{{
  "labels": [
    {{
      "label_index": <int starting from 1>,
      "top": <int 0-1000>,
      "left": <int 0-1000>,
      "bottom": <int 0-1000>,
      "right": <int 0-1000>,
      "category": "<empty|identifier|printed|handwritten|mixed>",
      "rotation_angle": <float>,
      "confidence": <float 0-1>
    }}
  ]
}}

Category definitions:
- "empty": blank label with no visible text or content
- "identifier": label whose PRIMARY content is a QR code, barcode, data-matrix, or
  machine-readable identifier. A label is still "identifier" even if it contains a small
  amount of supporting printed text such as a URI, catalogue number, or institution code
  alongside the code/barcode.
- "printed": label with only typed/printed text (no barcodes/QR codes as primary element)
- "handwritten": label with only handwritten text
- "mixed": label containing BOTH printed AND handwritten text in significant amounts

Bounding box rules:
- Use the 0–1000 normalised coordinate system described above.
- Make bounding boxes that FULLY COVER the entire label including its edges and any border.
  It is better to be slightly too large than to cut off part of a label.
- Include ALL visible labels, even very small ones.

rotation_angle: degrees to rotate CLOCKWISE to make the text horizontal and readable.
Use 0 if already correctly oriented.

Return ONLY valid JSON, no extra text."""


def _rescale_bbox(
    bbox: dict, img_w: int, img_h: int, padding_pct: float = 0.02,
) -> dict:
    """Convert a bounding box from Gemini's top/left/bottom/right 0-1000 coords
    to pixel-space xmin/ymin/xmax/ymax.

    The prompt uses field names that match Gemini's native y-first convention:
      top    → ymin (distance from top    edge, 0–1000)
      left   → xmin (distance from left   edge, 0–1000)
      bottom → ymax (distance from top    edge, 0–1000)
      right  → xmax (distance from left   edge, 0–1000)

    Adds a small padding (default 2 %) so the crop fully covers label edges.
    """
    raw_top = bbox.get("top", 0)
    raw_left = bbox.get("left", 0)
    raw_bottom = bbox.get("bottom", 1000)
    raw_right = bbox.get("right", 1000)

    pad_x = int(img_w * padding_pct)
    pad_y = int(img_h * padding_pct)
    return {
        "xmin": max(0, int(raw_left * img_w / 1000) - pad_x),
        "ymin": max(0, int(raw_top * img_h / 1000) - pad_y),
        "xmax": min(img_w, int(raw_right * img_w / 1000) + pad_x),
        "ymax": min(img_h, int(raw_bottom * img_h / 1000) + pad_y),
    }


def detect_and_classify(
    image_path: str,
    client: genai.Client,
    model: str = "gemini-2.5-flash",
) -> list[dict]:
    """
    Detect all labels in an image and classify them in a single API call.

    Gemini returns bounding boxes in 0-1000 normalised coordinates.
    This function converts them to pixel coordinates before returning.

    Args:
        image_path: Path to the image.
        client: Authenticated Gemini client.
        model: Gemini model to use.

    Returns:
        List of dicts with: label_index, xmin, ymin, xmax, ymax
        (in pixels), category, rotation_angle, confidence.
    """
    image_bytes, mime_type = _encode_image(image_path)

    # Read image dimensions for coordinate rescaling
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return []
    img_h, img_w = img.shape[:2]
    print(f"  Image size: {img_w}x{img_h}")

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            DETECTION_CLASSIFICATION_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    try:
        result = json.loads(response.text)
        labels = result.get("labels", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Warning: Failed to parse Gemini detection response: {e}")
        print(f"Raw response: {response.text[:500]}")
        return []

    # Log raw coordinates then convert to pixels
    for label in labels:
        raw = {k: label.get(k) for k in ("top", "left", "bottom", "right")}
        print(f"  Label {label.get('label_index')}: raw 0-1000 coords {raw}")
        pixel_bbox = _rescale_bbox(label, img_w, img_h)
        label.update(pixel_bbox)
        print(f"    → pixel coords {pixel_bbox}")

    return labels


# ---------------------Rotation---------------------#


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by the given angle (clockwise, in degrees).

    Args:
        image: Input image as numpy array.
        angle: Clockwise rotation angle in degrees.

    Returns:
        Rotated image as numpy array.
    """
    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # OpenCV uses counter-clockwise, so negate for clockwise
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # Calculate new bounding dimensions
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix for the new dimensions
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h), borderValue=(255, 255, 255)
    )


# ---------------------OCR / HTR---------------------#


OCR_PROMPT = """You are performing OCR (Optical Character Recognition) on an entomological specimen label image.

Transcribe ALL text visible in this image exactly as written. Preserve line breaks as spaces.
Do not interpret, translate, or correct the text — transcribe it verbatim.

If the image contains no readable text, return an empty string.

Return ONLY a JSON object:
{{
  "text": "<transcribed text>",
  "confidence": <float 0-1>
}}

Return ONLY valid JSON, no extra text."""


def ocr_image(
    image_path: str,
    client: genai.Client,
    model: str = "gemini-2.5-flash",
) -> dict:
    """
    Perform OCR/HTR on a single label image using Gemini.

    Works for printed, handwritten, and mixed labels.

    Args:
        image_path: Path to the label image.
        client: Authenticated Gemini client.
        model: Gemini model to use.

    Returns:
        Dict with keys: ID, text, confidence.
    """
    filename = Path(image_path).name
    image_bytes, mime_type = _encode_image(image_path)

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                OCR_PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        result = json.loads(response.text)
        return {
            "ID": filename,
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
        }

    except Exception as e:
        print(f"Error during Gemini OCR for {filename}: {e}")
        return {"ID": filename, "text": "ERROR", "confidence": 0.0}


def ocr_directory(
    image_dir: str,
    client: genai.Client,
    model: str = "gemini-2.5-flash",
    categories: Optional[list[str]] = None,
) -> list[dict]:
    """
    Perform OCR/HTR on all label images in a directory.

    Args:
        image_dir: Directory containing label images.
        client: Authenticated Gemini client.
        model: Gemini model to use.
        categories: If provided, only process images in these subdirectories
                    (e.g., ["printed", "handwritten", "mixed"]).

    Returns:
        List of dicts with keys: ID, text, confidence.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    results = []

    # Collect image files
    image_dir = Path(image_dir)
    if categories:
        # Process specific category subdirectories
        image_files = []
        for category in categories:
            cat_dir = image_dir / category
            if cat_dir.exists():
                image_files.extend(
                    f for f in sorted(cat_dir.iterdir())
                    if f.suffix.lower() in image_extensions
                    and not f.name.startswith("._")
                )
    else:
        # Process all images in the directory
        image_files = [
            f for f in sorted(image_dir.iterdir())
            if f.suffix.lower() in image_extensions
            and not f.name.startswith("._")
        ]

    if not image_files:
        print(f"No image files found in {image_dir}")
        return results

    print(f"Processing {len(image_files)} images with Gemini OCR...")
    for i, image_file in enumerate(image_files, 1):
        print(f"  [{i}/{len(image_files)}] {image_file.name}", end="\r")
        result = ocr_image(str(image_file), client, model)
        results.append(result)

    print()  # New line after progress
    return results
