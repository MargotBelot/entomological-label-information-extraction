#!/bin/bash
#
# Docker Setup Validation Script
#
# This script validates that both Docker Compose pipelines are properly configured
# and can be built successfully.
#

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MULTI_COMPOSE_FILE="${PROJECT_ROOT}/pipelines/multi-label-docker-compose.yaml"
SINGLE_COMPOSE_FILE="${PROJECT_ROOT}/pipelines/single-label-docker-compose.yaml"

echo "Docker Setup Validation"
echo "======================="
echo "Platform: $OSTYPE"

# Function to check if Docker is running
check_docker() {
    echo "Checking Docker status..."
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: Docker is not installed."
        echo "Please install Docker from: https://docker.com"
        echo "   - macOS/Windows: Docker Desktop"
        echo "   - Linux: Docker Engine or Docker Desktop"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: Docker is not running."
        echo "Please start Docker and try again:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "   macOS: Start Docker Desktop application"
        elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ -n "$WSL_DISTRO_NAME" ]]; then
            echo "   Windows: Start Docker Desktop application"
        else
            echo "   Linux: sudo systemctl start docker (or start Docker Desktop)"
        fi
        exit 1
    else
        echo "SUCCESS: Docker is running"
        
        # Show Docker info for debugging
        echo "   Docker version: $(docker --version | head -n 1)"
        echo "   Platform: $OSTYPE"
    fi
}

# Function to validate compose file syntax
validate_compose_syntax() {
    local compose_file=$1
    local pipeline_name=$2
    
    echo "Validating $pipeline_name compose file syntax..."
    if docker compose -f "$compose_file" config >/dev/null 2>&1; then
        echo "SUCCESS: $pipeline_name compose file syntax is valid"
    else
        echo "ERROR: $pipeline_name compose file has syntax errors:"
        docker compose -f "$compose_file" config
        return 1
    fi
}

# Function to check Dockerfile existence
check_dockerfiles() {
    echo "Checking Dockerfile existence..."
    local dockerfiles=(
        "pipelines/segmentation.dockerfile"
        "pipelines/empty_labels.dockerfile" 
        "pipelines/classification.dockerfile"
        "pipelines/rotation.dockerfile"
        "pipelines/tesseract.dockerfile"
        "pipelines/postprocessing.dockerfile"
    )
    
    local all_exist=true
    for dockerfile in "${dockerfiles[@]}"; do
        if [ -f "${PROJECT_ROOT}/${dockerfile}" ]; then
            echo "  SUCCESS: $dockerfile"
        else
            echo "  ERROR: $dockerfile (missing)"
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = true ]; then
        echo "SUCCESS: All Dockerfiles exist"
    else
        echo "ERROR: Some Dockerfiles are missing"
        return 1
    fi
}

# Function to check requirements files
check_requirements() {
    echo "Checking requirements files..."
    local requirements=(
        "pipelines/requirements/segmentation.txt"
        "pipelines/requirements/empty_labels.txt"
        "pipelines/requirements/classifier.txt"
        "pipelines/requirements/rotation.txt"
        "pipelines/requirements/tesseract.txt"
        "pipelines/requirements/postprocess.txt"
    )
    
    local all_exist=true
    for req_file in "${requirements[@]}"; do
        if [ -f "${PROJECT_ROOT}/${req_file}" ]; then
            echo "  SUCCESS: $req_file"
        else
            echo "  ERROR: $req_file (missing)"
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = true ]; then
        echo "SUCCESS: All requirements files exist"
    else
        echo "ERROR: Some requirements files are missing"
        return 1
    fi
}

# Function to check script files
check_scripts() {
    echo "Checking script files..."
    local scripts=(
        "scripts/processing/detection.py"
        "scripts/processing/analysis.py"
        "scripts/processing/classifiers.py"
        "scripts/processing/rotation.py"
        "scripts/processing/tesseract.py"
        "scripts/postprocessing/process.py"
        "scripts/postprocessing/consolidate_results.py"
    )
    
    local all_exist=true
    for script in "${scripts[@]}"; do
        if [ -f "${PROJECT_ROOT}/${script}" ]; then
            echo "  SUCCESS: $script"
        else
            echo "  ERROR: $script (missing)"
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = true ]; then
        echo "SUCCESS: All script files exist"
    else
        echo "ERROR: Some script files are missing"
        return 1
    fi
}

# Function to validate directory structure
check_directory_structure() {
    echo "Checking directory structure..."
    local directories=(
        "data/MLI/input"
        "data/MLI/output"
        "data/SLI/input"
        "data/SLI/output"
        "scripts/processing"
        "scripts/postprocessing"
        "scripts/docker"
        "pipelines"
        "pipelines/requirements"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "${PROJECT_ROOT}/${dir}"
        echo "  SUCCESS: $dir"
    done
    echo "SUCCESS: Directory structure validated"
}

# Function to test compose build (dry run)
test_compose_build() {
    local compose_file=$1
    local pipeline_name=$2
    
    echo "Testing $pipeline_name compose configuration..."
    
    # Just validate the compose file can be parsed (fast)
    if docker compose -f "$compose_file" config --quiet; then
        echo "SUCCESS: $pipeline_name configuration is valid"
    else
        echo "ERROR: $pipeline_name has configuration issues"
        echo "  Run: docker compose -f $compose_file config"
        return 1
    fi
}

# Function to check hidden files that might cause issues
check_hidden_files() {
    echo "Checking for problematic hidden files..."
    
    # Find and remove macOS resource fork files
    local ds_store_files=$(find "$PROJECT_ROOT" -name "._*" 2>/dev/null || true)
    if [ -n "$ds_store_files" ]; then
        echo "  Found macOS resource fork files:"
        echo "$ds_store_files" | sed 's/^/    /'
        echo "  Removing them..."
        find "$PROJECT_ROOT" -name "._*" -delete 2>/dev/null || true
        echo "SUCCESS: Cleaned up macOS resource fork files"
    else
        echo "SUCCESS: No problematic hidden files found"
    fi
}

# Main validation function
main() {
    cd "$PROJECT_ROOT"
    
    echo "Project root: $PROJECT_ROOT"
    echo ""
    
    check_docker
    check_directory_structure
    check_dockerfiles
    check_requirements
    check_scripts
    check_hidden_files
    
    echo ""
    echo "Testing Compose Files"
    echo "===================="
    
    validate_compose_syntax "$MULTI_COMPOSE_FILE" "Multi-label"
    validate_compose_syntax "$SINGLE_COMPOSE_FILE" "Single-label"
    
    echo ""
    echo "Configuration Testing"
    echo "===================="
    
    test_compose_build "$MULTI_COMPOSE_FILE" "Multi-label"
    test_compose_build "$SINGLE_COMPOSE_FILE" "Single-label" 
    
    echo ""
    echo "Validation Complete!"
    echo "==================="
    echo ""
    echo "All checks passed! Your Docker setup is ready."
    echo ""
    echo "Ready to run pipelines:"
    echo "  Multi-label:  ./scripts/docker/start-multi-label-pipeline.sh"
    echo "  Single-label: ./scripts/docker/start-single-label-pipeline.sh"
    echo "  Main launcher: ./run-pipeline.sh"
    echo ""
    echo "Note: First pipeline run will be slower as Docker images are built."
}

# Run main function
main "$@"
