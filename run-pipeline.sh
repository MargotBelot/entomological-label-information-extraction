#!/bin/bash
#
# Entomological Label Processing Pipeline Launcher
#
# This script provides an easy way to launch either the multi-label or single-label
# processing pipeline with automatic Docker management and cleanup.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Entomological Label Information Extraction"
echo "==========================================="
echo ""
echo "Choose your option:"
echo ""
echo "1) Multi-Label Pipeline  - Process full specimen images (includes detection)"
echo "   Input:  Full specimen images in data/MLI/input/"
echo "   Output: data/MLI/output/consolidated_results.json"
echo ""
echo "2) Single-Label Pipeline - Process pre-cropped label images"  
echo "   Input:  Pre-cropped images in data/SLI/input/"
echo "   Output: data/SLI/output/consolidated_results.json"
echo ""
echo "3) Validate Setup        - Check environment and Docker configuration"
echo "   Recommended for first-time users or troubleshooting"
echo ""
echo "4) Exit"
echo ""

while true; do
    read -p "Enter your choice (1-4): " choice
    case $choice in
        1)
            echo ""
            echo "Launching Multi-Label Pipeline..."
            exec "${SCRIPT_DIR}/scripts/docker/start-multi-label-pipeline.sh"
            ;;
        2)
            echo ""
            echo "Launching Single-Label Pipeline..."
            exec "${SCRIPT_DIR}/scripts/docker/start-single-label-pipeline.sh"
            ;;
        3)
            echo ""
            echo "Validating Docker Setup..."
            exec "${SCRIPT_DIR}/scripts/docker/validate-docker-setup.sh"
            ;;
        4)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter 1, 2, 3, or 4."
            ;;
    esac
done
