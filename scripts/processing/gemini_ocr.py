#!/usr/bin/env python3
"""
Gemini OCR / HTR Script

Performs text recognition on label images using the Gemini API.
Unlike Tesseract and Google Vision which only handle printed text,
Gemini can process printed, handwritten, and mixed labels.

Output format matches the existing pipeline: JSON list of {ID, text, confidence}.

Usage:
    python gemini_ocr.py -d <image_dir> -o <output_dir>
    python gemini_ocr.py -d <output_dir> -o <output_dir> --categories printed handwritten mixed
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from label_processing.gemini_processor import get_client, ocr_directory
from label_processing import utils


# Output filename (matches existing convention)
FILENAME = "ocr_gemini.json"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform OCR/HTR on label images using Gemini API.",
    )
    parser.add_argument(
        "-d", "--dir", type=str, required=True,
        help="Directory containing label images to process.",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory for OCR results JSON.",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Process only images in these subdirectories "
             "(e.g., --categories printed handwritten mixed). "
             "If not set, processes all images in the directory.",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Gemini API key. Falls back to GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash).",
    )
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_arguments()

    if not os.path.exists(args.dir):
        print(f"Error: Input directory '{args.dir}' does not exist.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # Initialize Gemini client
    print("Initializing Gemini client...")
    client = get_client(args.api_key)

    # Run OCR on the directory
    print(f"Running Gemini OCR on {args.dir}")
    if args.categories:
        print(f"  Categories: {', '.join(args.categories)}")

    results = ocr_directory(
        image_dir=args.dir,
        client=client,
        model=args.model,
        categories=args.categories,
    )

    # Save results
    if results:
        utils.save_json(results, FILENAME, args.outdir)
        print(f"\nOCR results saved to {os.path.join(args.outdir, FILENAME)}")
        print(f"Processed {len(results)} images")

        # Quick stats
        non_empty = sum(1 for r in results if r.get("text", "").strip())
        errors = sum(1 for r in results if r.get("text") == "ERROR")
        print(f"  Non-empty results: {non_empty}")
        if errors:
            print(f"  Errors: {errors}")
    else:
        print("No images were processed.")

    duration = time.time() - start_time
    print(f"Finished in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
