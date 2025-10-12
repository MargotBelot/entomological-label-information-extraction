#!/bin/bash

set -e

# Get script directory and project root - portable approach
if [ -n "${BASH_SOURCE[0]}" ]; then
    # Bash
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -n "$0" ]; then
    # POSIX shell fallback
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
else
    # Last resort
    SCRIPT_DIR="$(pwd)"
fi

PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Running Complete SLI Pipeline in Single Container..."
echo "===================================================="
echo "Single-Label Information extraction for pre-cropped label images"
echo ""

# Run everything in one container to avoid volume mounting issues
docker run --rm --pid=host --ipc=host --ulimit nofile=65536:65536 \
    -v "$PROJECT_ROOT/data:/data" \
    -v "$PROJECT_ROOT/models:/models" \
    pipelines-detection \
    bash -c "
        echo '=== Step 1: Detection (Label Validation) ==='
        python3 scripts/processing/detection.py -j data/SLI/input -o data/SLI/output --batch-size 1 --device cpu
        
        echo ''
        echo '=== Step 2: Empty/Not-Empty Classification ==='
        python3 scripts/processing/analysis.py -o data/SLI/output -i data/SLI/output/input_cropped
        
        echo ''  
        echo '=== Step 3: ID/Description Classification ==='
        python3 scripts/processing/classifiers.py -m 1 -j data/SLI/output/not_empty -o data/SLI/output
        
        echo ''
        echo '=== Step 4: Handwritten/Printed Classification ==='
        python3 scripts/processing/classifiers.py -m 2 -j data/SLI/output/not_identifier -o data/SLI/output
        
        echo ''
        echo '=== Step 5: OCR with Tesseract ==='
        python3 scripts/processing/tesseract.py -d data/SLI/output/printed -o data/SLI/output
        
        echo ''
        echo '=== Step 6: Post-processing ==='
        python3 scripts/postprocessing/process.py -j data/SLI/output/ocr_preprocessed.json -o data/SLI/output
        python3 scripts/postprocessing/consolidate_results.py -o data/SLI/output -f consolidated_results.json
        
        echo ''
        echo '=== Final Results ==='
        ls -la data/SLI/output/
        echo ''
        echo 'Results files:'
        find data/SLI/output/ -name '*.json' -o -name '*.csv' | head -10
        
        echo ''
        echo '=== Copying results to persistent location ==='
        cp -r data/SLI/output/* /data/SLI/output/ || true
        chown -R $(id -u):$(id -g) /data/SLI/output/ 2>/dev/null || true
        
        echo '=== SLI PIPELINE COMPLETED ==='
    "

echo ""
echo "SLI Pipeline finished! Results in: $PROJECT_ROOT/data/SLI/output/"