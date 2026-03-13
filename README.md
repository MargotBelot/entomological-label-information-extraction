# Entomological Label Information Extraction (ELIE)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-stable-brightgreen)
[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue)](https://entomological-label-information-extraction.readthedocs.io/en/latest/)
[![Tests](https://img.shields.io/badge/tests-passing-success)](unit_tests/)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey)
[![Docker](https://img.shields.io/badge/docker-supported-2496ED?logo=docker&logoColor=white)](pipelines/Dockerfile)
[![Conda](https://img.shields.io/badge/conda-supported-44A833?logo=anaconda&logoColor=white)](environment.yml)

**AI-powered text extraction from insect specimen labels using computer vision and OCR**

Extract and digitize text from museum specimen labels automatically — including handwritten text. Perfect for museum digitization, research data preparation, and biodiversity informatics.

> **Related Repository**: For advanced clustering and deduplication of extracted labels, see [ELIE-clustering](https://github.com/joel-tuberosa/ELIE-clustering)

---

## What It Does

- **Input**: Specimen photos (full images) or pre-cropped label images
- **Process**: Detects labels → classifies → corrects orientation → extracts text (OCR/HTR) → cleans output → extracts structured entities → validates against GBIF & OSM
- **Output**: Structured JSON/CSV files with extracted text, Darwin Core records, and quality reports

### Three Pipeline Types

| Pipeline | Input | Detection | Classification & OCR | Best For |
|----------|-------|-----------|---------------------|----------|
| **Gemini** (recommended) | Full specimen photos or pre-cropped labels | Gemini API | Gemini API | Printed + handwritten labels; no local model needed |
| **MLI** (traditional) | Full specimen photos | Detectron2 (local) | TensorFlow + Tesseract | Printed labels only; offline use |
| **SLI** (traditional) | Pre-cropped labels | Not needed | TensorFlow + Tesseract | Printed labels only; offline use |

---

## Quick Start

### 1. Install Prerequisites

**a) Conda** — Python package manager ([Install Miniconda](https://conda.io/miniconda.html))
```bash
conda --version
```

**b) Tesseract OCR** — only needed for the traditional pipelines
```bash
# macOS:
brew install tesseract

# Linux (Ubuntu/Debian):
sudo apt install tesseract-ocr

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Install ELIE

```bash
git clone https://github.com/MargotBelot/entomological-label-information-extraction.git
cd entomological-label-information-extraction

conda env create -f environment.yml
conda activate ELIE
pip install -e .
```

### 3. Set Up API Key (Gemini Pipeline)

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey), then:
```bash
export GEMINI_API_KEY=<your-api-key>
```

### 4. Add Your Images

```bash
# Full specimen photos → Gemini pipeline will detect labels automatically
cp /path/to/your/photos/*.jpg data/MLI/input/
```

### 5. Run the Pipeline

**Option A: Streamlit GUI (Recommended)**
```bash
python launch.py
```
Opens a web interface where you can select a pipeline, configure options, review results, correct OCR, and re-run entity recognition — all from the browser.

**Option B: Command Line**
```bash
# Gemini pipeline (recommended)
./tools/pipelines/run_gemini_pipeline_conda.sh

# Traditional MLI pipeline
./tools/pipelines/run_mli_pipeline_conda.sh

# Traditional SLI pipeline
./tools/pipelines/run_sli_pipeline_conda.sh
```

You can override defaults with environment variables:
```bash
INPUT_DIR=data/MLI/input OUTPUT_DIR=data/MLI/output \
ENTITY_RECOGNITION=true EXPORT_DWC=true EXPORT_CSV=true \
./tools/pipelines/run_gemini_pipeline_conda.sh
```

### 6. View Results

Results saved to the output directory:

| File | Description |
|------|-------------|
| `entity_master.json` | All labels with extracted entities, GBIF validation, OSM geocoding |
| `consolidated_results.json` | Labels with OCR text and metadata |
| `quality_report.json` | Extraction quality scores per label |
| `darwin_core.json` | Darwin Core formatted records (one per specimen) |
| `darwin_core.csv` | Same as above in CSV format |
| `validated_results.json` | After manual OCR corrections in the Streamlit UI |

---

## Gemini Pipeline Workflow

The Gemini pipeline replaces local models with the Gemini API for all vision tasks:

```
Specimen Image
    │
    ▼
┌─────────────────────────────┐
│ Step 1+2: Gemini Detection  │  Gemini finds all labels, classifies them
│   + Classification          │  (printed/handwritten/mixed/identifier/empty),
│   + Rotation                │  and determines rotation angle
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 3: OCR / HTR           │  Gemini reads text from each label
│   (Gemini, Tesseract, or   │  (works for printed AND handwritten)
│    Google Vision)           │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 4: Post-processing     │  Text cleaning, consolidation
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 5: Entity Recognition  │  Gemini extracts structured entities;
│   + GBIF + OSM enrichment   │  validates names with GBIF; geocodes
│                             │  localities with OpenStreetMap
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Step 6+7: Crop & Cleanup    │  Optional label cropping; remove
│                             │  intermediate files
└─────────────────────────────┘
```

---

## Traditional Pipeline Workflow

For offline processing using local models (Detectron2, TensorFlow, Tesseract):

1. **Detection** (MLI only): Detect and crop labels from specimen photos
2. **Classification**: Filter empty labels, identify QR codes, classify text type
3. **Rotation**: Correct label orientation for better OCR
4. **OCR**: Extract text using Tesseract (or Google Vision API)
5. **Post-processing**: Clean and structure extracted text
6. **Output**: Consolidated JSON with all results and metadata

---

## Streamlit Web Interface

The built-in Streamlit interface (`python launch.py`) provides:

- **Pipeline configuration**: Select pipeline type, OCR engine, entity recognition options
- **Real-time progress**: Live processing dashboard with logs and metrics
- **Label explorer**: Browse images with annotated bounding boxes
- **OCR correction**: Edit transcribed text directly in the browser
- **Entity viewer**: See extracted scientific names, collectors, dates, geography
- **Re-run entity recognition**: After correcting OCR text, re-extract entities with one click
- **Download**: Export all outputs (JSON, CSV, Darwin Core)

---

## Docker (Optional)

For containerized execution:

```bash
cd pipelines

# Gemini pipeline (lightweight — no local models needed)
GEMINI_API_KEY=<your-key> docker-compose --profile gemini up

# Traditional MLI pipeline
docker-compose --profile mli up

# Traditional SLI pipeline
docker-compose --profile sli up
```

See `pipelines/README.md` for complete Docker documentation.

---

## Project Structure

```
entomological-label-information-extraction/
├── data/
│   ├── MLI/input/                    # Full specimen photos
│   ├── MLI/output/                   # Results
│   └── SLI/input/                    # Pre-cropped labels
├── label_processing/                 # Core processing modules
│   ├── gemini_processor.py           # Gemini detection, classification, OCR
│   ├── entity_recognition.py         # Entity extraction, GBIF, OSM, DwC/OpenDS
│   ├── label_detection.py            # Detectron2 detection (traditional)
│   ├── tensorflow_classifier.py      # TF classifiers (traditional)
│   └── ...
├── scripts/
│   ├── processing/
│   │   ├── gemini_classify.py        # Gemini detection + classification CLI
│   │   ├── gemini_ocr.py             # Gemini OCR CLI
│   │   ├── entity_recognition.py     # Entity recognition CLI
│   │   ├── detection.py              # Detectron2 detection CLI
│   │   └── ...
│   ├── postprocessing/
│   │   ├── consolidate_results.py    # Merge pipeline outputs
│   │   └── crop_labels.py            # Crop labels from originals
│   └── evaluation/                   # Analysis tools
├── interfaces/
│   └── launch_streamlit.py           # Streamlit web GUI
├── pipelines/
│   ├── Dockerfile                    # Docker multi-stage build
│   ├── docker-compose.yml            # Docker orchestration
│   └── requirements/                 # Per-stage dependencies
├── tools/
│   ├── pipelines/                    # Shell scripts (Gemini, MLI, SLI)
│   └── hpc/                          # SLURM job templates
├── unit_tests/                       # Test suite
├── launch.py                         # Quick launcher (Streamlit)
├── environment.yml                   # Conda environment
└── pyproject.toml                    # Python package config
```

---

## Troubleshooting

### Common Issues

**"conda activate ELIE" fails**
```bash
# Ensure environment was created:
conda env create -f environment.yml
# Restart terminal, then:
conda activate ELIE
```

**"GEMINI_API_KEY not set"**
```bash
export GEMINI_API_KEY=<your-api-key>
# Get a key from https://aistudio.google.com/apikey
```

**"ModuleNotFoundError"**
```bash
conda activate ELIE
pip install -e .
```

**"No images found"**
```bash
# Check images are in the correct folder:
ls data/MLI/input/   # Should show image files (.jpg, .png, .tiff)
```

**"Tesseract not found"** (traditional pipeline only)
```bash
# macOS:
brew install tesseract
# Linux:
sudo apt install tesseract-ocr
```

**Still stuck?**
- Full docs: https://entomological-label-information-extraction.readthedocs.io/
- Run diagnostic: `python scripts/health_check.py`

---

## Technical Details

### Models

| Component | Gemini Pipeline | Traditional Pipeline |
|-----------|----------------|---------------------|
| Detection | Gemini 2.5 Flash (API) | Faster R-CNN / Detectron2 (local) |
| Classification | Gemini 2.5 Flash (API) | TensorFlow CNNs (local) |
| OCR / HTR | Gemini 2.5 Flash (API) | Tesseract or Google Vision |
| Entity Recognition | Gemini 2.0 Flash (API) + GBIF + OSM | Not available |
| Post-processing | NLTK for text cleaning | NLTK for text cleaning |

### Accuracy

- **Label Detection**: 94% accuracy (IoU ≥ 0.8)
- **Classification**: 98-100% accuracy
- **Rotation Correction**: 97.3% accuracy
- **OCR (printed)**: Median CER 0.0-5.0%, WER 0.0-22%

### Resource Requirements

| Pipeline | RAM | CPU | GPU | Internet |
|----------|-----|-----|-----|----------|
| Gemini | 2 GB | Any | Not needed | Required (API) |
| Traditional MLI | 6 GB | 4+ cores | Optional | Not needed |
| Traditional SLI | 4 GB | 2+ cores | Optional | Not needed |

---

## Citation

```bibtex
@article{belot2026high,
  author = {Belot, Margot and Tuberosa, J. and Preuss, Leonardo and Svezhentseva, Olha and Claessen, Magdalena and B{\"o}lling, C. and Schuster, F. and L{\'e}ger, T.},
  title = {High-throughput information extraction of printed specimen labels from large-scale digitization of entomological collections using a semi-automated pipeline},
  journal = {Methods in Ecology and Evolution},
  year = {2026},
  doi = {10.1111/2041-210x.70235},
  url = {https://doi.org/10.1111/2041-210x.70235}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) file

---

## Documentation

**Full documentation:** https://entomological-label-information-extraction.readthedocs.io/

### Quick Links
- [Quick Start Guide](https://entomological-label-information-extraction.readthedocs.io/en/latest/quickstart.html)
- [User Guide](https://entomological-label-information-extraction.readthedocs.io/en/latest/user_guide.html)
- [API Reference](https://entomological-label-information-extraction.readthedocs.io/en/latest/api/modules.html)

### Additional Guides
- [Advanced Configuration](docs/ADVANCED_CONFIG.md)
- [Rotation Model Setup](docs/ROTATION_MODEL_SETUP.md)
- [Docker README](pipelines/README.md)

### Related Projects
- [ELIE-clustering](https://github.com/joel-tuberosa/ELIE-clustering) — Advanced clustering for label deduplication and similarity analysis
