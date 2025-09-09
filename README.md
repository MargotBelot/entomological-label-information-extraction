# Entomological Label Information Extraction

**AI-powered text extraction from insect specimen labels**

Automatically extract and digitize text from museum specimen labels using artificial intelligence. Process thousands of specimens.

## Table of Contents
- [Entomological Label Information Extraction](#entomological-label-information-extraction)
  - [Table of Contents](#table-of-contents)
  - [What This Tool Does](#what-this-tool-does)
  - [Prerequisites](#prerequisites)
  - [Try It Right Now (2 minutes)](#try-it-right-now-2-minutes)
  - [Need Help Getting Started?](#need-help-getting-started)
  - [Using Your Own Images](#using-your-own-images)
  - [Understanding the Results](#understanding-the-results)
  - [Want to Learn More?](#want-to-learn-more)
  - [Technical Details](#technical-details)
  - [Sample Data and Training](#sample-data-and-training)

## What This Tool Does

**The Problem:** Museums have millions of insect specimens with handwritten and printed labels containing valuable scientific data, but manual transcription is extremely time-consuming.

**The Solution:** This AI system automatically:
- **Finds labels** in specimen photos
- **Reads the text** using computer vision
- **Organizes the data** into spreadsheets
- **Processes thousands** of specimens quickly

**Why It Works:** AI models specifically trained on entomological data with high accuracy and reproducible results.

## Prerequisites

**Before you start, you need:**

1. **Docker Desktop** (recommended - easiest setup)
   - Download from [docker.com](https://docker.com)
   - Make sure it's running before starting the pipeline
   - Allocate 8GB+ RAM to Docker in settings

**Alternative:** Manual Python setup
   - Python 3.9+ with conda or pip
   - See [TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) for detailed installation

**System requirements:**
- 8GB+ RAM, 2GB+ disk space
- Works on Linux, macOS, Windows

## Try It Right Now (2 minutes)

**Make sure you have [Docker Desktop running first](#prerequisites)!**

**Step 1:** Get the code and run it
```bash
git clone https://github.com/MargotBelot/entomological-label-information-extraction.git
cd entomological-label-information-extraction
./run-pipeline.sh
```

**Step 2:** Choose your pipeline when prompted:
- **Multi-Label Pipeline** - for full specimen photos
- **Single-Label Pipeline** - for pre-cropped label images

**Step 3:** Check your results
```bash
ls data/MLI/output/                    # Multi-Label results
ls data/SLI/output/                    # Single-Label results  
```

**What you'll get:** Individual label images, specimen IDs, and all text extracted into structured JSON and CSV files.

## Need Help Getting Started?

**Problem installing or running?** Common solutions:

- **"Docker not found"** - Install Docker Desktop from docker.com
- **"Permission denied"** - Run `chmod +x run-pipeline.sh`
- **"No output files"** - Make sure your images are in .jpg format
- **Pipeline stuck** - Check Docker Desktop has 8GB+ RAM allocated

**Still having issues?** See [TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) for detailed troubleshooting.

## Using Your Own Images

**Ready to process your specimen collection?** Here's how:

**For full specimen photos:**
1. Put your .jpg images in `data/MLI/input/`
2. Run `./run-pipeline.sh` and choose "Multi-Label Pipeline"
3. Results appear in `data/MLI/output/consolidated_results.json`

**For pre-cropped label images:**
1. Put your .jpg images in `data/SLI/input/`
2. Run `./run-pipeline.sh` and choose "Single-Label Pipeline"
3. Results appear in `data/SLI/output/consolidated_results.json`

**Best image quality:** High resolution (300+ DPI), clear lighting, .jpg format only.

## Understanding the Results

**After processing, you'll find these key files:**

- **`consolidated_results.json`** - Main result file linking all processing stages for each image
- **`input_cropped/`** - Individual label images found and extracted
- **`identifier.csv`** - Catalog numbers and specimen IDs
- **`printed/`, `handwritten/`** - Labels sorted by text type for further processing

**The processing automatically:**
1. Detects and crops labels from specimen photos  
2. Classifies them as empty/useful, identifier/descriptive, handwritten/printed
3. Applies rotation correction (Single-Label pipeline only)
4. Extracts text using OCR
5. Cleans and structures the data

## Want to Learn More?

**Complete guides available:**

- **[USER_GUIDE.md](docs/USER_GUIDE.md)** - Detailed usage with examples and FAQ
- **[DOCKER_SETUP.md](docs/DOCKER_SETUP.md)** - Docker setup and advanced options
- **[TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md)** - Installation, troubleshooting, development setup

## Technical Details

**System Requirements:**
- Docker Desktop (recommended) OR Python 3.9+ with manual setup
- 8GB+ RAM, 2GB+ disk space  
- Cross-platform: Linux, macOS, Windows

**Key Features:**
- PyTorch 2.6+ compatible with automatic GPU detection
- Model caching for 50-90% faster subsequent runs
- Environment independent - works from any directory
- Automatic fallbacks for CPU/GPU processing

**Pipeline Differences:**
- **Multi-Label:** Full specimen photos → label detection → classification → OCR
- **Single-Label:** Pre-cropped labels → classification → rotation correction → OCR

## Sample Data and Training

**Included sample data:**
- `data/MLI/` - Multi-label specimen images (ready to test)
- `data/SLI/` - Single-label images (ready to test)

**Training datasets:** Available on Zenodo at [https://doi.org/10.7479/khac-x956](https://doi.org/10.7479/khac-x956)

**Model retraining:** See `training_notebooks/` for Jupyter notebooks

---

**License:** MIT - see [LICENSE](LICENSE) file  
**Issues:** Report bugs on GitHub  
**Contributing:** See TECHNICAL_GUIDE.md for development setup
