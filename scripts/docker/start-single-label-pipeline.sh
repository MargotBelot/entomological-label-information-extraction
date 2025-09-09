#!/bin/bash
#
# Single-Label Pipeline Starter Script
#
# This script automatically:
# 1. Ensures Docker is running
# 2. Cleans up any previous pipeline runs
# 3. Starts the single-label processing pipeline
# 4. Monitors the pipeline execution
#

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/pipelines/single-label-docker-compose.yaml"

echo "Starting Single-Label Entomological Label Processing Pipeline"
echo "======================================================"

# Function to check if Docker is running
check_docker() {
    echo "Checking Docker status..."
    
    # Check if Docker is installed
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: Docker is not installed."
        echo "Please install Docker from: https://docker.com"
        echo "   - macOS/Windows: Docker Desktop"
        echo "   - Linux: Docker Engine or Docker Desktop"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo "WARNING: Docker is not running. Attempting to start..."
        
        # Try to start Docker Desktop on macOS
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if [ -d "/Applications/Docker.app" ]; then
                echo "   Starting Docker Desktop on macOS..."
                open -a Docker
                echo "Waiting for Docker to start..."
                
                # Wait for Docker to be ready (max 60 seconds)
                local count=0
                while ! docker info >/dev/null 2>&1; do
                    if [ $count -ge 60 ]; then
                        echo "ERROR: Docker failed to start within 60 seconds"
                        echo "Please start Docker Desktop manually and run this script again"
                        exit 1
                    fi
                    sleep 2
                    count=$((count + 2))
                done
                echo "SUCCESS: Docker is now running"
            else
                echo "ERROR: Docker Desktop not found in /Applications/"
                echo "Please install Docker Desktop or start it manually"
                exit 1
            fi
        else
            echo "ERROR: Cannot auto-start Docker on this platform."
            echo "Please start Docker manually:"
            if [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ -n "$WSL_DISTRO_NAME" ]]; then
                echo "   Windows: Start Docker Desktop application"
            else
                echo "   Linux: sudo systemctl start docker (or start Docker Desktop)"
            fi
            echo "Then run this script again."
            exit 1
        fi
    else
        echo "SUCCESS: Docker is running"
    fi
}

# Function to cleanup previous runs
cleanup_previous_runs() {
    echo "Cleaning up previous pipeline runs..."
    
    # Stop any running containers from previous runs
    if docker ps --format "table {{.Names}}" | grep -E "(detection|classifier|tesseract|postprocessing)" >/dev/null 2>&1; then
        echo "  Stopping running containers..."
        docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    fi
    
    # Remove any stopped containers from previous runs  
    if docker ps -a --format "table {{.Names}}" | grep -E "(detection|classifier|tesseract|postprocessing)" >/dev/null 2>&1; then
        echo "  Removing stopped containers..."
        docker compose -f "$COMPOSE_FILE" rm -f 2>/dev/null || true
    fi
    
    # Clean up any orphaned volumes (optional - comment out if you want to keep data)
    # docker volume prune -f 2>/dev/null || true
    
    echo "SUCCESS: Cleanup complete"
}

# Function to create necessary directories
create_directories() {
    echo "Creating necessary directories..."
    mkdir -p "${PROJECT_ROOT}/data/SLI/input"
    mkdir -p "${PROJECT_ROOT}/data/SLI/output"
    echo "SUCCESS: Directories created"
}

# Function to validate input directory
validate_input() {
    local input_dir="${PROJECT_ROOT}/data/SLI/input"
    
    echo "Validating input directory..."
    if [ ! -d "$input_dir" ] || [ -z "$(ls -A "$input_dir" 2>/dev/null)" ]; then
        echo "WARNING: Input directory is empty: $input_dir"
        echo "Please add your image files to process before running the pipeline."
        echo "Supported formats: .jpg, .jpeg, .png"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        local file_count=$(find "$input_dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
        echo "SUCCESS: Found $file_count image files to process"
    fi
}

# Function to monitor pipeline progress
monitor_pipeline() {
    echo "Monitoring pipeline progress..."
    echo "Press Ctrl+C to stop monitoring (pipeline will continue running)"
    echo "======================================================"
    
    # Follow logs from all services
    docker compose -f "$COMPOSE_FILE" logs -f
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    check_docker
    cleanup_previous_runs
    create_directories
    validate_input
    
    echo "Building and starting pipeline services..."
    echo "This may take several minutes on first run..."
    
    # Build and run the pipeline
    if docker compose -f "$COMPOSE_FILE" up --build; then
        echo "SUCCESS: Pipeline completed successfully!"
        echo "Results are available in: ${PROJECT_ROOT}/data/SLI/output/"
        echo "Consolidated results: ${PROJECT_ROOT}/data/SLI/output/consolidated_results.json"
    else
        echo "ERROR: Pipeline failed with errors"
        echo "Check the logs above for details"
        exit 1
    fi
}

# Handle script interruption
trap 'echo "Pipeline monitoring interrupted. Pipeline continues in background."; exit 0' INT

# Run main function
main "$@"
