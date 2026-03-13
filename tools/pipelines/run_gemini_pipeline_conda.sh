#!/bin/bash

set -e

# Get script directory and project root
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -n "$0" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
else
    SCRIPT_DIR="$(pwd)"
fi

PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

echo "Running Gemini Pipeline with Conda Environment..."
echo "=================================================="
echo "Uses Gemini API for classification, rotation, and optionally OCR/HTR"
echo ""

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ELIE

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ---------- Configuration ----------
# Input/output paths (can be overridden via environment variables)
INPUT_DIR=${INPUT_DIR:-"data/MLI/input"}
OUTPUT_DIR=${OUTPUT_DIR:-"data/MLI/output"}

# OCR engine: "gemini", "tesseract", or "vision"
OCR_ENGINE=${OCR_ENGINE:-"gemini"}

# Pipeline mode: "MLI" (full specimen images) or "SLI" (pre-cropped labels)
PIPELINE_MODE=${PIPELINE_MODE:-"MLI"}

# Gemini API key (required - set via env var or pass as argument)
# GEMINI_API_KEY should be set in the environment

# Google Vision credentials (only needed if OCR_ENGINE=vision)
# GOOGLE_VISION_CREDENTIALS should be set in the environment

echo "Pipeline mode: $PIPELINE_MODE"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "OCR engine: $OCR_ENGINE"

# Validate Gemini API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set."
    echo "Set it with: export GEMINI_API_KEY=<your-api-key>"
    exit 1
fi

# ---------- Step 1+2: Gemini Detection + Classification + Rotation ----------
echo ""
echo "=== Step 1+2: Gemini Detection + Classification + Rotation ==="

CLASSIFY_ARGS="-i $INPUT_DIR -o $OUTPUT_DIR"
if [ "$PIPELINE_MODE" = "MLI" ]; then
    echo "Using Gemini for label detection (no local model needed)"
    CLASSIFY_ARGS="$CLASSIFY_ARGS --gemini-detection"
else
    echo "SLI mode: each image treated as a single label"
fi

python scripts/processing/gemini_classify.py $CLASSIFY_ARGS || echo "Gemini detection+classification completed with warnings"

# ---------- Step 3: OCR / HTR ----------
echo ""
echo "=== Step 3: OCR ($OCR_ENGINE) ==="

# Determine which directories have images to process
# Gemini classification saves images into category subdirectories
if [ "$OCR_ENGINE" = "gemini" ]; then
    # Gemini OCR: can process printed, handwritten, and mixed labels
    python scripts/processing/gemini_ocr.py \
        -d "$OUTPUT_DIR" \
        -o "$OUTPUT_DIR" \
        --categories printed handwritten mixed \
        || echo "Gemini OCR completed with warnings"
    OCR_JSON="$OUTPUT_DIR/ocr_gemini.json"

elif [ "$OCR_ENGINE" = "tesseract" ]; then
    # Tesseract OCR: only process printed labels
    if [ -d "$OUTPUT_DIR/printed" ] && [ -n "$(ls -A "$OUTPUT_DIR/printed" 2>/dev/null)" ]; then
        python scripts/processing/tesseract.py -d "$OUTPUT_DIR/printed" -o "$OUTPUT_DIR" || echo "Tesseract OCR completed with warnings"
    else
        echo "Warning: No printed labels found for Tesseract OCR"
    fi
    OCR_JSON="$OUTPUT_DIR/ocr_preprocessed.json"

elif [ "$OCR_ENGINE" = "vision" ]; then
    # Google Vision OCR: only process printed labels
    if [ -z "$GOOGLE_VISION_CREDENTIALS" ]; then
        echo "Error: GOOGLE_VISION_CREDENTIALS environment variable is not set."
        echo "Set it with: export GOOGLE_VISION_CREDENTIALS=/path/to/credentials.json"
        exit 1
    fi
    if [ -d "$OUTPUT_DIR/printed" ] && [ -n "$(ls -A "$OUTPUT_DIR/printed" 2>/dev/null)" ]; then
        python scripts/processing/vision.py -c "$GOOGLE_VISION_CREDENTIALS" -d "$OUTPUT_DIR/printed" -o "$OUTPUT_DIR" || echo "Google Vision OCR completed with warnings"
    else
        echo "Warning: No printed labels found for Google Vision OCR"
    fi
    OCR_JSON="$OUTPUT_DIR/ocr_google_vision.json"

else
    echo "Error: Unknown OCR engine '$OCR_ENGINE'. Use: gemini, tesseract, or vision"
    exit 1
fi

# ---------- Step 4: Post-processing ----------
echo ""
echo "=== Step 4: Post-processing ==="
if [ -f "$OCR_JSON" ]; then
    python scripts/postprocessing/process.py -j "$OCR_JSON" -o "$OUTPUT_DIR" || echo "Post-processing completed with warnings"
    python scripts/postprocessing/consolidate_results.py -o "$OUTPUT_DIR" -f consolidated_results.json || echo "Consolidation completed with warnings"
else
    echo "Warning: No OCR results found at $OCR_JSON"
fi

# ---------- Step 5: Entity Recognition (optional) ----------
ENTITY_RECOGNITION=${ENTITY_RECOGNITION:-"false"}
echo ""
echo "=== Step 5: Entity Recognition ==="
if [ "$ENTITY_RECOGNITION" = "true" ]; then
    CONSOLIDATED="$OUTPUT_DIR/consolidated_results.json"
    if [ -f "$CONSOLIDATED" ]; then
        ENTITY_ARGS="-i $CONSOLIDATED -o $OUTPUT_DIR"
        [ "${EXPORT_DWC:-false}" = "true" ] && ENTITY_ARGS="$ENTITY_ARGS --dwc"
        [ "${EXPORT_OPENDS:-false}" = "true" ] && ENTITY_ARGS="$ENTITY_ARGS --opends"
        [ "${EXPORT_CSV:-false}" = "true" ] && ENTITY_ARGS="$ENTITY_ARGS --csv"
        python scripts/processing/entity_recognition.py $ENTITY_ARGS || echo "Entity recognition completed with warnings"
    else
        echo "  Skipped: consolidated_results.json not found"
    fi
else
    echo "  Skipped (not enabled)"
fi

# ---------- Step 6: Crop labels (optional) ----------
echo ""
echo "=== Step 6: Crop Labels ==="
CROP_LABELS=${CROP_LABELS:-"false"}
if [ "$CROP_LABELS" = "true" ]; then
    if [ -f "$OUTPUT_DIR/entity_master.json" ] || [ -f "$OUTPUT_DIR/consolidated_results.json" ]; then
        python scripts/postprocessing/crop_labels.py -i "$INPUT_DIR" -o "$OUTPUT_DIR" || echo "Cropping completed with warnings"
    else
        echo "  Skipped: no JSON source found for bounding boxes"
    fi
else
    echo "  Skipped (not enabled)"
fi

# ---------- Step 7: Clean up intermediate files ----------
echo ""
echo "=== Step 7: Cleanup ==="
# Remove intermediate processing files
for f in \
    "$OUTPUT_DIR/gemini_classification.json" \
    "$OUTPUT_DIR/gemini_classification.csv" \
    "$OUTPUT_DIR/corrected_transcripts.json" \
    "$OUTPUT_DIR/plausible_transcripts.json" \
    "$OUTPUT_DIR/empty_transcripts.csv" \
    "$OUTPUT_DIR/identifier.csv"; do
    [ -f "$f" ] && rm "$f" && echo "  Removed $(basename $f)"
done
# Remove detection CSV (dynamically named)
for f in "$OUTPUT_DIR"/*_predictions.csv; do
    [ -f "$f" ] && rm "$f" && echo "  Removed $(basename $f)"
done
# Remove detection crops directory (intermediate — cropped_labels/ is the final one)
for d in "$OUTPUT_DIR"/*_cropped; do
    [ -d "$d" ] && rm -rf "$d" && echo "  Removed $(basename $d)/"
done
# If entity_master.json exists, consolidated_results.json is redundant
if [ -f "$OUTPUT_DIR/entity_master.json" ] && [ -f "$OUTPUT_DIR/consolidated_results.json" ]; then
    rm "$OUTPUT_DIR/consolidated_results.json" && echo "  Removed consolidated_results.json (superseded by entity_master.json)"
fi
echo "  Final outputs kept:" && find "$OUTPUT_DIR" -maxdepth 1 \( -name "*.json" -o -name "*.csv" \) -exec basename {} \; 2>/dev/null | sort | sed 's/^/    /'

# ---------- Summary ----------
echo ""
echo "=== Final Results ==="
echo "Output files:"
find "$OUTPUT_DIR/" -maxdepth 1 \( -name "*.json" -o -name "*.csv" \) 2>/dev/null

echo ""
echo "Final output captured"
echo "✅ Gemini pipeline completed successfully!"
