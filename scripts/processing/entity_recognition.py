#!/usr/bin/env python3
"""
Entity Recognition CLI Script

Reads consolidated_results.json, runs LLM-based entity extraction with GBIF/OSM
enrichment, and writes:
  - entity_master.json   (always)
  - quality_report.json  (always)
  - darwin_core.json     (optional --dwc)
  - open_ds.json         (optional --opends)
  - darwin_core.csv      (optional --csv)

Usage:
    python entity_recognition.py -i <consolidated_results.json> -o <output_dir>
    python entity_recognition.py -i <file> -o <dir> --dwc --opends --csv
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from label_processing.gemini_processor import get_client
from label_processing.entity_recognition import (
    extract_and_enrich,
    validate_and_normalize,
    generate_dwc,
    generate_opends,
    export_to_csv,
    build_master_json,
)
from label_processing import utils


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract structured biodiversity entities from OCR output.",
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to consolidated_results.json.",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory for entity recognition results.",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Gemini API key. Falls back to GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.0-flash",
        help="Gemini model to use (default: gemini-2.0-flash).",
    )
    parser.add_argument(
        "--dwc", action="store_true",
        help="Export Darwin Core JSON.",
    )
    parser.add_argument(
        "--opends", action="store_true",
        help="Export OpenDS JSON.",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Export Darwin Core records as CSV.",
    )
    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_arguments()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # Load consolidated results
    print(f"Loading consolidated results from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        consolidated = json.load(f)
    print(f"  Found {len(consolidated)} label records")

    # Initialize Gemini client
    print("Initializing Gemini client...")
    client = get_client(args.api_key)

    # Step 1: Extract entities + GBIF/OSM enrichment
    print(f"\n=== Entity Extraction (model: {args.model}) ===")
    enriched = extract_and_enrich(consolidated, client, args.model)

    # Step 2: Validate & score quality
    print("\n=== Validation & Quality Scoring ===")
    validated, quality_report = validate_and_normalize(enriched)

    # Step 3: Build & save master JSON (always)
    master = build_master_json(validated, quality_report)
    master_path = os.path.join(args.outdir, "entity_master.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved master JSON → {master_path}")

    # Step 4: Save quality report (always)
    report_path = os.path.join(args.outdir, "quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    print(f"  Saved quality report → {report_path}")

    # Step 5: Optional exports
    if args.dwc:
        dwc_records = generate_dwc(validated)
        dwc_path = os.path.join(args.outdir, "darwin_core.json")
        with open(dwc_path, "w", encoding="utf-8") as f:
            json.dump(dwc_records, f, indent=2, ensure_ascii=False)
        print(f"  Saved DwC JSON → {dwc_path} ({len(dwc_records)} records)")

        if args.csv:
            csv_path = os.path.join(args.outdir, "darwin_core.csv")
            export_to_csv(dwc_records, csv_path)

    if args.opends:
        opends_records = generate_opends(validated)
        opends_path = os.path.join(args.outdir, "open_ds.json")
        with open(opends_path, "w", encoding="utf-8") as f:
            json.dump(opends_records, f, indent=2, ensure_ascii=False)
        print(f"  Saved OpenDS JSON → {opends_path} ({len(opends_records)} records)")

    # Summary
    duration = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"  Labels processed: {len(validated)}")
    for line in quality_report.get("summary", []):
        print(f"  {line}")
    if quality_report.get("overall_extraction_rate"):
        print(f"  Overall extraction rate: {quality_report['overall_extraction_rate']}")
    print(f"  Finished in {duration:.2f} seconds")


if __name__ == "__main__":
    main()
