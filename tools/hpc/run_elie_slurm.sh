#!/bin/bash
#SBATCH --job-name=elie-pipeline
#SBATCH --output=elie-%j.out
#SBATCH --error=elie-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard

# ELIE Pipeline - SLURM Job Script Template
# 
# Usage:
#   1. Edit the configuration variables below
#   2. Submit: sbatch run_elie_slurm.sh
#   3. Monitor: squeue -u $USER
#   4. View output: tail -f elie-<jobid>.out

# ============================================
# CONFIGURATION - EDIT THESE VARIABLES
# ============================================

# Path to Apptainer container (.sif file)
CONTAINER_PATH="/path/to/elie.sif"

# Data directories on HPC
INPUT_DATA_DIR="/scratch/$USER/elie/input"
OUTPUT_DATA_DIR="/scratch/$USER/elie/output"

# Pipeline selection: "mli" or "sli"
PIPELINE_TYPE="mli"

# ============================================
# PIPELINE EXECUTION (DO NOT EDIT BELOW)
# ============================================

echo "=========================================="
echo "ELIE Pipeline - HPC Execution"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Pipeline: $PIPELINE_TYPE"
echo "Input: $INPUT_DATA_DIR"
echo "Output: $OUTPUT_DATA_DIR"
echo "=========================================="
echo ""

# Load Apptainer module (adjust for your HPC system)
if command -v module &> /dev/null; then
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
fi

# Check if container exists
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "ERROR: Container not found at $CONTAINER_PATH"
    echo "Please build the container first:"
    echo "  cd pipelines"
    echo "  apptainer build elie.sif elie.def"
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DATA_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DATA_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DATA_DIR"

# Run the pipeline
echo "Starting pipeline execution..."
echo ""

INPUT_DIR=/app/data/input OUTPUT_DIR=/app/data/output \
    apptainer run \
    --bind $INPUT_DATA_DIR:/app/data/input \
    --bind $OUTPUT_DATA_DIR:/app/data/output \
    $CONTAINER_PATH $PIPELINE_TYPE

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Pipeline completed successfully!"
    echo "Results saved to: $OUTPUT_DATA_DIR"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Pipeline failed with errors"
    echo "Check the error log: elie-${SLURM_JOB_ID}.err"
    echo "=========================================="
    exit 1
fi
