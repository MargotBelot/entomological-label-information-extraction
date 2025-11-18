#!/usr/bin/env python3
"""
Native Pipeline Runner for Entomological Label Information Extraction

This script runs the complete pipeline without Docker, using the trained models
directly on the local system.

Usage:
    python run_pipeline_native.py --pipeline mli --input data/MLI/input --output data/MLI/output
    python run_pipeline_native.py --pipeline sli --input data/SLI/input --output data/SLI/output
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Native pipeline runner for entomological label processing."""
    
    def __init__(self, input_dir: str, output_dir: str, pipeline_type: str = "mli"):
        """
        Initialize the pipeline runner.
        
        Args:
            input_dir: Path to input images
            output_dir: Path to output directory
            pipeline_type: Type of pipeline ('mli' or 'sli')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.pipeline_type = pipeline_type.lower()
        self.project_root = Path(__file__).parent
        
        # Validate inputs
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized: {self.pipeline_type.upper()}")
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
    
    def run_command(self, cmd: list, step_name: str) -> bool:
        """
        Run a command and handle errors.
        
        Args:
            cmd: Command to run as list
            step_name: Name of the step for logging
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Starting {step_name}...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"{step_name} completed successfully in {elapsed:.2f}s")
            
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.debug(f"STDERR: {result.stderr}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"{step_name} failed with return code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error running {step_name}: {e}")
            return False
    
    def run_mli_pipeline(self) -> bool:
        """Run the complete MLI (Multi-Label) pipeline."""
        logger.info("Running MLI (Multi-Label) Pipeline")
        
        steps = [
            # Step 1: Label Detection
            {
                "name": "Label Detection",
                "cmd": [
                    "python", "processing/detection.py",
                    "-j", str(self.input_dir),
                    "-o", str(self.output_dir),
                    "--device", "cpu",
                    "--batch-size", "1"
                ]
            },
            
            # Step 2: Empty Label Classification
            {
                "name": "Empty Label Classification",
                "cmd": [
                    "python", "processing/analysis.py",
                    "-o", str(self.output_dir),
                    "-i", str(self.output_dir / f"{os.path.basename(self.input_dir)}_cropped")
                ]
            },
            
            # Step 3: Identifier/Description Classification
            {
                "name": "Identifier Classification",
                "cmd": [
                    "python", "processing/classifiers.py",
                    "-m", "1",
                    "-j", str(self.output_dir / "not_empty"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 4: Handwritten/Printed Classification
            {
                "name": "Text Type Classification",
                "cmd": [
                    "python", "processing/classifiers.py",
                    "-m", "2",
                    "-j", str(self.output_dir / "not_identifier"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 5: Rotation Correction
            {
                "name": "Rotation Correction",
                "cmd": [
                    "python", "processing/rotation.py",
                    "-i", str(self.output_dir / "printed"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 6: OCR Processing
            {
                "name": "OCR Processing",
                "cmd": [
                    "python", "processing/tesseract.py",
                    "-d", str(self.output_dir / "rotated"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 7: Post-processing
            {
                "name": "Post-processing",
                "cmd": [
                    "python", "postprocessing/process.py",
                    "-j", str(self.output_dir / "ocr_preprocessed.json"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 8: Consolidate Results
            {
                "name": "Consolidate Results",
                "cmd": [
                    "python", "postprocessing/consolidate_results.py",
                    "-o", str(self.output_dir),
                    "-f", "consolidated_results.json"
                ]
            }
        ]
        
        for step in steps:
            if not self.run_command(step["cmd"], step["name"]):
                logger.error(f"Pipeline failed at step: {step['name']}")
                return False
        
        logger.info("MLI Pipeline completed successfully!")
        return True
    
    def run_sli_pipeline(self) -> bool:
        """Run the SLI (Single-Label) pipeline."""
        logger.info("Running SLI (Single-Label) Pipeline")
        
        steps = [
            # Step 1: Empty Label Classification
            {
                "name": "Empty Label Classification",
                "cmd": [
                    "python", "processing/analysis.py",
                    "-o", str(self.output_dir),
                    "-i", str(self.input_dir)
                ]
            },
            
            # Step 2: Identifier/Description Classification
            {
                "name": "Identifier Classification",
                "cmd": [
                    "python", "processing/classifiers.py",
                    "-m", "1",
                    "-j", str(self.output_dir / "not_empty"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 3: Handwritten/Printed Classification
            {
                "name": "Text Type Classification",
                "cmd": [
                    "python", "processing/classifiers.py",
                    "-m", "2",
                    "-j", str(self.output_dir / "not_identifier"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 4: Rotation Correction
            {
                "name": "Rotation Correction",
                "cmd": [
                    "python", "processing/rotation.py",
                    "-i", str(self.output_dir / "printed"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 5: OCR Processing
            {
                "name": "OCR Processing",
                "cmd": [
                    "python", "processing/tesseract.py",
                    "-d", str(self.output_dir / "rotated"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 6: Post-processing
            {
                "name": "Post-processing",
                "cmd": [
                    "python", "postprocessing/process.py",
                    "-j", str(self.output_dir / "ocr_preprocessed.json"),
                    "-o", str(self.output_dir)
                ]
            },
            
            # Step 7: Consolidate Results
            {
                "name": "Consolidate Results",
                "cmd": [
                    "python", "postprocessing/consolidate_results.py",
                    "-o", str(self.output_dir),
                    "-f", "consolidated_results.json"
                ]
            }
        ]
        
        for step in steps:
            if not self.run_command(step["cmd"], step["name"]):
                logger.error(f"Pipeline failed at step: {step['name']}")
                return False
        
        logger.info("SLI Pipeline completed successfully!")
        return True
    
    def run_pipeline(self) -> bool:
        """Run the appropriate pipeline based on type."""
        if self.pipeline_type == "mli":
            return self.run_mli_pipeline()
        elif self.pipeline_type == "sli":
            return self.run_sli_pipeline()
        else:
            logger.error(f"Unknown pipeline type: {self.pipeline_type}")
            return False
    
    def print_results_summary(self):
        """Print a summary of the results."""
        results_file = self.output_dir / "consolidated_results.json"
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                logger.info(f"\n{'='*60}")
                logger.info("PIPELINE RESULTS SUMMARY")
                logger.info(f"{'='*60}")
                logger.info(f"Total files processed: {len(results)}")
                
                # Count by classification
                empty_count = sum(1 for r in results if r.get('classification', {}).get('empty', False))
                identifier_count = sum(1 for r in results if r.get('classification', {}).get('identifier', False))
                handwritten_count = sum(1 for r in results if r.get('classification', {}).get('handwritten', False))
                printed_count = sum(1 for r in results if r.get('classification', {}).get('printed', False))
                
                logger.info(f"Empty labels: {empty_count}")
                logger.info(f"Identifiers (QR/barcode): {identifier_count}")
                logger.info(f"Handwritten labels: {handwritten_count}")
                logger.info(f"Printed labels: {printed_count}")
                
                # OCR results
                ocr_count = sum(1 for r in results if r.get('ocr', {}).get('raw_text'))
                logger.info(f"Labels with OCR text: {ocr_count}")
                
                logger.info(f"\nMain results file: {results_file}")
                logger.info(f"Output directory: {self.output_dir}")
                
            except Exception as e:
                logger.error(f"Error reading results summary: {e}")
        else:
            logger.warning("Results file not found - pipeline may have failed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run Entomological Label Pipeline (Native Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MLI pipeline (multi-label)
  python run_pipeline_native.py --pipeline mli --input data/MLI/input --output data/MLI/output
  
  # Run SLI pipeline (single-label)
  python run_pipeline_native.py --pipeline sli --input data/SLI/input --output data/SLI/output
  
  # Run with custom paths
  python run_pipeline_native.py --pipeline mli --input /path/to/images --output /path/to/results
        """
    )
    
    parser.add_argument(
        "--pipeline",
        choices=["mli", "sli"],
        required=True,
        help="Pipeline type: 'mli' for multi-label images, 'sli' for single-label images"
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing images"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize and run pipeline
        runner = PipelineRunner(args.input, args.output, args.pipeline)
        
        start_time = time.time()
        success = runner.run_pipeline()
        elapsed_time = time.time() - start_time
        
        if success:
            logger.info(f"\nPipeline completed successfully in {elapsed_time:.2f} seconds")
            runner.print_results_summary()
            sys.exit(0)
        else:
            logger.error("Pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

