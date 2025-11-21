#!/usr/bin/env python3
"""
Native Pipeline Runner for Entomological Label Information Extraction.

This script runs the complete pipeline without Docker, using the trained models
directly on the local system.

Usage:
    python run_full_pipeline.py --pipeline mli --input data/MLI/input --output data/MLI/output
    python run_full_pipeline.py --pipeline sli --input data/SLI/input --output data/SLI/output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

# Add project root to Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import pipeline step functions
from scripts.processing.analysis import run_empty_label_classification
from scripts.processing.classifiers import run_classification
from scripts.processing.detection import run_label_detection
from scripts.processing.rotation import run_rotation_correction
from scripts.processing.tesseract import run_ocr_with_tesseract
from scripts.postprocessing.consolidate_results import run_consolidate_results
from scripts.postprocessing.process import process_ocr_output

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
)
logger = logging.getLogger(__name__)

PipelineType = Literal["mli", "sli"]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a single pipeline run."""

    pipeline_type: PipelineType
    input_dir: Path
    output_dir: Path


StepFunc = Callable[..., None]
Step = tuple[str, StepFunc, tuple[Any, ...], dict[str, Any]]


def validate_and_build_config(
    pipeline: str,
    input_dir: str,
    output_dir: str,
) -> PipelineConfig:
    """Validate CLI arguments and return a PipelineConfig."""
    normalized = pipeline.lower()
    if normalized not in ("mli", "sli"):
        msg = f"Invalid pipeline type: {pipeline}. Must be 'mli' or 'sli'"
        raise ValueError(msg)

    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        msg = f"Input directory not found: {in_path}"
        raise FileNotFoundError(msg)

    out_path.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        pipeline_type=normalized,  # type: ignore[assignment]
        input_dir=in_path,
        output_dir=out_path,
    )

    logger.info(f"Pipeline initialized: {cfg.pipeline_type.upper()}")
    logger.info(f"Input: {cfg.input_dir}")
    logger.info(f"Output: {cfg.output_dir}")

    return cfg


def run_step(
    step_name: str,
    func: StepFunc,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Execute a pipeline step with error handling and timing."""
    logger.info(f"Starting {step_name}...")
    start_time = time.time()

    try:
        func(*args, **kwargs)
    except Exception:
        logger.exception(f"{step_name} failed")
        return False

    elapsed = time.time() - start_time
    logger.info(f"{step_name} completed successfully in {elapsed:.2f}s")
    return True


def get_cropped_dir(cfg: PipelineConfig) -> Path:
    """Get the path to the cropped images directory."""
    return cfg.output_dir / f"{cfg.input_dir.name}_cropped"


def postprocess_ocr(cfg: PipelineConfig) -> None:
    """Post-process OCR results for a given config."""
    ocr_json = cfg.output_dir / "ocr_preprocessed.json"
    if not ocr_json.exists():
        msg = f"OCR results file not found: {ocr_json}"
        raise FileNotFoundError(msg)

    process_ocr_output(ocr_output=str(ocr_json), outdir=str(cfg.output_dir))


def consolidate_results(cfg: PipelineConfig) -> None:
    """Consolidate all pipeline results for a given config."""
    run_consolidate_results(
        output_dir=str(cfg.output_dir),
        filename="consolidated_results.json",
    )


def build_mli_steps(cfg: PipelineConfig) -> list[Step]:
    """Build the list of steps for the MLI (Multi-Label) pipeline."""
    cropped_dir = get_cropped_dir(cfg)
    not_empty_dir = cfg.output_dir / "not_empty"
    not_identifier_dir = cfg.output_dir / "not_identifier"
    printed_dir = cfg.output_dir / "printed"

    return [
        (
            "Label Detection",
            run_label_detection,
            (),
            {
                "input_dir": str(cfg.input_dir),
                "input_image": None,
                "output_dir": str(cfg.output_dir),
                "confidence_threshold": 0.8,
                "batch_size": 1,
                "device": "cpu",
                "no_cache": False,
                "clear_cache": False,
            },
        ),
        (
            "Empty Label Classification",
            run_empty_label_classification,
            (),
            {
                "input_image_dir": str(cropped_dir),
                "output_image_dir": str(cfg.output_dir),
            },
        ),
        (
            "Identifier Classification",
            run_classification,
            (),
            {
                "jpg_dir": str(not_empty_dir),
                "out_dir": str(cfg.output_dir),
                "model": 1,
            },
        ),
        (
            "Text Type Classification",
            run_classification,
            (),
            {
                "jpg_dir": str(not_identifier_dir),
                "out_dir": str(cfg.output_dir),
                "model": 2,
            },
        ),
        (
            "OCR Processing",
            run_ocr_with_tesseract,
            (),
            {
                "crop_dir": str(printed_dir),
                "outdir": str(cfg.output_dir),
                "thresholding": 1,
                "blocksize": None,
                "c_value": None,
                "multiprocessing": False,
                "verbose": False,
            },
        ),
        ("Post-processing", postprocess_ocr, (cfg,), {}),
        ("Consolidate Results", consolidate_results, (cfg,), {}),
    ]


def build_sli_steps(cfg: PipelineConfig) -> list[Step]:
    """Build the list of steps for the SLI (Single-Label) pipeline."""
    not_empty_dir = cfg.output_dir / "not_empty"
    not_identifier_dir = cfg.output_dir / "not_identifier"
    printed_dir = cfg.output_dir / "printed"
    rotated_dir = cfg.output_dir / "rotated"

    return [
        (
            "Empty Label Classification",
            run_empty_label_classification,
            (),
            {
                "input_image_dir": str(cfg.input_dir),
                "output_image_dir": str(cfg.output_dir),
            },
        ),
        (
            "Identifier Classification",
            run_classification,
            (),
            {
                "jpg_dir": str(not_empty_dir),
                "out_dir": str(cfg.output_dir),
                "model": 1,
            },
        ),
        (
            "Text Type Classification",
            run_classification,
            (),
            {
                "jpg_dir": str(not_identifier_dir),
                "out_dir": str(cfg.output_dir),
                "model": 2,
            },
        ),
        (
            "Rotation Correction",
            run_rotation_correction,
            (),
            {
                "input_image_dir": str(printed_dir),
                "output_image_dir": str(rotated_dir),
            },
        ),
        (
            "OCR Processing",
            run_ocr_with_tesseract,
            (),
            {
                "crop_dir": str(rotated_dir),
                "outdir": str(cfg.output_dir),
                "thresholding": 1,
                "blocksize": None,
                "c_value": None,
                "multiprocessing": False,
                "verbose": False,
            },
        ),
        ("Post-processing", postprocess_ocr, (cfg,), {}),
        ("Consolidate Results", consolidate_results, (cfg,), {}),
    ]


def run_pipeline(cfg: PipelineConfig) -> bool:
    """Run the appropriate pipeline based on config."""
    if cfg.pipeline_type == "mli":
        logger.info("Running MLI (Multi-Label) Pipeline")
        steps = build_mli_steps(cfg)
    elif cfg.pipeline_type == "sli":
        logger.info("Running SLI (Single-Label) Pipeline")
        steps = build_sli_steps(cfg)
    else:
        logger.error(f"Unknown pipeline type: {cfg.pipeline_type}")
        return False

    for name, func, args, kwargs in steps:
        if not run_step(name, func, *args, **kwargs):
            logger.error(f"Pipeline failed at step: {name}")
            return False

    logger.info(f"{cfg.pipeline_type.upper()} Pipeline completed successfully!")
    return True


def print_results_summary(cfg: PipelineConfig) -> None:
    """Print a summary of the results."""
    results_file = cfg.output_dir / "consolidated_results.json"

    if not results_file.exists():
        logger.warning("Results file not found - pipeline may have failed")
        return

    try:
        with results_file.open("r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception:
        logger.exception("Error reading results summary")
        return

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len(results)}")

    empty_count = sum(
        1 for r in results if r.get("classification", {}).get("empty", False)
    )
    identifier_count = sum(
        1 for r in results if r.get("classification", {}).get("identifier", False)
    )
    handwritten_count = sum(
        1 for r in results if r.get("classification", {}).get("handwritten", False)
    )
    printed_count = sum(
        1 for r in results if r.get("classification", {}).get("printed", False)
    )

    logger.info(f"Empty labels: {empty_count}")
    logger.info(f"Identifiers (QR/barcode): {identifier_count}")
    logger.info(f"Handwritten labels: {handwritten_count}")
    logger.info(f"Printed labels: {printed_count}")

    ocr_count = sum(1 for r in results if r.get("ocr", {}).get("raw_text"))
    logger.info(f"Labels with OCR text: {ocr_count}")

    logger.info(f"\nMain results file: {results_file}")
    logger.info(f"Output directory: {cfg.output_dir}")


def run_full_pipeline(
    pipeline: str,
    input_dir: str,
    output_dir: str,
    verbose: bool = False,
) -> int:
    """
    Run the full pipeline end-to-end and return an exit code.
    
    Args:
        pipeline (str): The pipeline type to run.
        input_dir (str): The input directory containing the images.
        output_dir (str): The output directory for the results.
        verbose (bool): Whether to enable verbose logging.

    Returns:
        int: The exit code of the pipeline.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cfg = validate_and_build_config(
            pipeline=pipeline,
            input_dir=input_dir,
            output_dir=output_dir,
        )

        start_time = time.time()
        success = run_pipeline(cfg)
        elapsed_time = time.time() - start_time

        if success:
            logger.info(
                f"\nPipeline completed successfully in {elapsed_time:.2f} seconds",
            )
            print_results_summary(cfg)
            return 0

        logger.error("Pipeline failed!")
        return 1

    except Exception:
        logger.exception("Pipeline error")
        return 1


def main() -> None:
    """Parse command-line arguments and run the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Entomological Label Pipeline (Native Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Run MLI pipeline (multi-label)\n"
            "  python run_full_pipeline.py --pipeline mli --input data/MLI/input --output data/MLI/output\n"
            "\n"
            "  # Run SLI pipeline (single-label)\n"
            "  python run_full_pipeline.py --pipeline sli --input data/SLI/input --output data/SLI/output\n"
            "\n"
            "  # Run with custom paths\n"
            "  python run_full_pipeline.py --pipeline mli --input /path/to/images --output /path/to/results\n"
        ),
    )

    parser.add_argument(
        "--pipeline",
        choices=["mli", "sli"],
        required=True,
        help="Pipeline type: 'mli' for multi-label images, 'sli' for single-label images",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    exit_code = run_full_pipeline(
        pipeline=args.pipeline,
        input_dir=args.input,
        output_dir=args.output,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
