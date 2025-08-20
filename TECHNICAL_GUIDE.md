# Technical Guide

This guide covers installation, system setup, troubleshooting, and advanced technical topics.

## System Requirements

**Minimum Requirements:**
- Python 3.9+
- 8GB RAM
- 2GB free disk space
- macOS 10.14+ or Linux (Ubuntu 18.04+, CentOS 7+)

**Recommended for Large Collections:**
- Python 3.10+
- 16GB+ RAM
- GPU with 4GB+ VRAM
- SSD storage

## Installation

**Installation Options Summary:**
- `pip install -e .` - Basic installation (runtime dependencies only)
- `pip install -e .[dev]` - Development installation (includes testing, linting, formatting tools)
- `pip install -e ".[test]"` - Testing installation (includes pytest and coverage tools)
- `pip install -e ".[docs]"` - Documentation installation (includes Sphinx and themes)
- `pip install -e ".[dev,test,docs]"` - Complete installation (all optional dependencies)

**Step 1: Clone the Repository**

```bash
git clone https://github.com/[your-username]/entomological-label-information-extraction.git
cd entomological-label-information-extraction
```

**Step 2: Python Environment Setup**

**Option A: Conda (Recommended)**

# Install conda if needed:
# Mac: https://docs.anaconda.com/anaconda/install/mac-os/
# Linux: https://docs.anaconda.com/anaconda/install/linux/

An up-to-date `environment.yml` is provided in the repository.  
To set up all dependencies, simply run:

```bash
conda env create -f environment.yml
conda activate entomological-label
```

There is no need to create a new environment file—just use the one provided.

**Install the package (basic installation):**
```bash
pip install -e .
```

**OR for development (includes testing, linting, formatting tools):**
```bash
pip install -e .[dev]
```

**Option B: Virtual Environment**

```bash
# Create virtual environment
python3 -m venv elie-env

# Activate environment
source elie-env/bin/activate  # Mac/Linux

# Install the package (basic installation)
pip install -e .

# OR for development (includes testing, linting, formatting tools)
pip install -e .[dev]
```

**Step 3: System Dependencies**

**macOS:**

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install tesseract zbar
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt update
sudo apt install tesseract-ocr libzbar0 python3-dev python3-pip
```

**Linux (CentOS/RHEL/Fedora):**

```bash
# CentOS/RHEL
sudo yum install tesseract zbar-devel python3-devel

# Fedora
sudo dnf install tesseract zbar-devel python3-devel
```

**Step 4: Verify Installation**

```bash
# Test package installation
python3 -c "import label_processing; print('✅ Package installed successfully')"

# Test system dependencies
tesseract --version
python3 -c "import cv2; print('✅ OpenCV working')"

# Run test suite
python3 -m pytest unit_tests/ -v
```

## Docker Installation

**Prerequisites:**
- Docker Desktop installed and running
- 8GB+ RAM allocated to Docker

**Quick Start:**

```bash
# Clone repository
git clone https://github.com/[your-username]/entomological-label-information-extraction.git
cd entomological-label-information-extraction

# Run multi-label pipeline (includes sample data)
docker compose -f multi-label-docker-compose.yaml up --build

# Results will be in data/MLI/output/
```

**Available Docker Compose Configurations:**

- `multi-label-docker-compose.yaml` - Full pipeline with detection
- `single-label-docker-compose.yaml` - Pipeline for pre-cropped labels

## Troubleshooting

### Installation Issues

**Problem: "No module named 'label_processing'"**

```bash
# Solution: Reinstall the package
cd /path/to/entomological-label-information-extraction
pip install -e .

# For development with all tools
pip install -e .[dev]
```

**Problem: "TesseractNotFoundError"**

```bash
# macOS solution
brew install tesseract

# Linux solution
sudo apt install tesseract-ocr    # Ubuntu/Debian
sudo yum install tesseract         # CentOS/RHEL
```

**Problem: "Command 'gcc' failed" or compilation errors**

```bash
# macOS: Install Xcode command line tools
xcode-select --install

# Linux: Install build tools
sudo apt install build-essential python3-dev  # Ubuntu/Debian
```

### Model Loading Issues

**Problem: "invalid load key, 'v'" or "Model loading failed"**

This is typically a CUDA/CPU compatibility issue. The improved model loading code in the label_processing package automatically handles this by trying multiple loading strategies:

- Normal PyTorch loading
- CPU fallback with map_location='cpu' 
- weights_only mode for newer PyTorch versions

This should resolve cross-platform model loading issues automatically.

**Problem: "CUDA out of memory"**

```bash
# Force CPU usage
python3 scripts/processing/detection.py -j images/ -o results/ --device cpu

# Or reduce batch size
python3 scripts/processing/detection.py -j images/ -o results/ --batch-size 4
```

### Runtime Issues

**Problem: "Processing is very slow"**

```bash
# Reduce batch size for limited RAM
--batch-size 4

# Use GPU if available
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Problem: "No labels detected"**

```bash
# Lower confidence threshold
--confidence 0.3

# Verify image quality and format
file your_image.jpg
```

**Problem: "Too many false detections"**

```bash
# Increase confidence threshold
--confidence 0.8
```

### Docker Issues

**Problem: "Docker daemon is not running"**

```bash
# Start Docker Desktop application
# Or on Linux:
sudo systemctl start docker
```

**Problem: "Out of memory" in Docker**

```bash
# Increase Docker memory allocation:
# Docker Desktop → Settings → Resources → Memory (8GB+)
```

**Problem: "Permission denied" with Docker**

```bash
# Linux: Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

## GPU Support

**NVIDIA GPU Setup:**

1. **Install NVIDIA drivers** for your GPU

2. **Install CUDA toolkit** (version 11.8 recommended):
   
   ```bash
   # Check if CUDA is available
   nvidia-smi
   ```

3. **Install PyTorch with CUDA support:**
   
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify GPU detection:**
   
   ```bash
   python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

**Docker GPU Support:**

1. **Install NVIDIA Container Toolkit**
2. **Use GPU-enabled Docker commands** (if available in your setup)

## Model Architecture Details

**Label Detection Model:**
- **Framework:** PyTorch
- **Architecture:** YOLO-based object detection
- **Input:** RGB images (any size, automatically resized)
- **Output:** Bounding boxes with confidence scores
- **Training Data:** Annotated entomological specimen images

**Classification Models:**
- **Framework:** TensorFlow
- **Purpose:** Categorize label types (handwritten vs. printed)
- **Input:** Cropped label images
- **Output:** Classification probabilities

**Text Recognition:**
- **Primary:** Tesseract OCR (offline, free)
- **Optional:** Google Cloud Vision API (requires setup and billing)

## Advanced Configuration

**Environment Variables:**

```bash
# Set custom model paths
export DETECTION_MODEL_PATH="/path/to/custom/model.pth"
export TESSERACT_CMD="/usr/local/bin/tesseract"

# OCR language configuration
export TESSERACT_LANG="eng+fra+deu"  # Multiple languages
```

**Custom Model Training:**

The repository includes Jupyter notebooks for retraining models:

- `training_notebooks/Label_Detection_Detecto_Training_Notebook.ipynb`
- `training_notebooks/Classifier_TensorFlow_Training_Notebook.ipynb`
- `training_notebooks/Label_Rotation_TensorFlow_Training_Notebook.ipynb`

**Google Cloud Vision API Setup:**

1. Create Google Cloud account
2. Enable Vision API
3. Download credentials JSON
4. Set environment variable:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   ```

5. Run Vision OCR:

   ```bash
   python3 scripts/processing/vision.py -d cropped_images/ -c credentials.json -o results/
   ```

## Development Setup

**For Contributors:**

```bash
# Clone and install in development mode
git clone https://github.com/your-fork/entomological-label-information-extraction.git
cd entomological-label-information-extraction

# Install with development dependencies
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install

# Run tests
python3 -m pytest unit_tests/ -v

# Run with coverage
python3 -m pytest unit_tests/ --cov=. --cov-report=html
```

**Code Quality Tools:**

```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .
```

**Testing:**

```bash
# Run all tests
python3 -m pytest unit_tests/ -v

# Run specific test modules
python3 -m pytest unit_tests/label_processing_tests/

# Run with coverage
python3 -m pytest unit_tests/ --cov=label_processing --cov-report=html
```

## API Reference

**Core Classes:**

```python
from label_processing.label_detection import PredictLabel
from label_processing.text_recognition import ImageProcessor, Tesseract

# Label Detection
detector = PredictLabel(
    path_to_model="models/detection_model.pth",
    classes=["label"],
    device="auto"  # 'cpu', 'cuda', or 'auto'
)

results = detector.class_prediction("image.jpg")

# OCR Processing
processor = ImageProcessor.read_image("label.jpg")
ocr = Tesseract(image=processor, language="eng")
text = ocr.image_to_string()
```

**Command Line Tools:**

```bash
# Main detection script
python3 scripts/processing/detection.py [options]

# Classification
python3 scripts/processing/classifiers.py [options]

# OCR processing
python3 scripts/processing/tesseract.py [options]

# Google Vision OCR
python3 scripts/processing/vision.py [options]
```

## Performance Optimization

**Memory Usage:**

- **Reduce batch size** for limited RAM systems
- **Process in chunks** for very large collections
- **Monitor system resources** during processing

**Processing Speed:**

- **Use GPU** for faster inference (if available)
- **Adjust image resolution** if speed is more important than accuracy
- **Parallel processing** for multiple independent batches

**Quality vs. Speed Trade-offs:**

```bash
# High quality (slower)
--confidence 0.9 --batch-size 8

# Balanced (default)
--confidence 0.5 --batch-size 16

# Fast processing (may miss some labels)
--confidence 0.3 --batch-size 32
```

## File Formats and Compatibility

**Supported Input Formats:**
- JPEG (.jpg, .jpeg) - Recommended
- PNG (.png) - Supported
- TIFF (.tiff, .tif) - Supported

**Output Formats:**
- CSV files for tabular data
- Cropped JPEG images for individual labels
- Processing logs in text format

**Compatibility:**
- **macOS:** 10.14+ (Mojave and newer)
- **Linux:** Ubuntu 18.04+, CentOS 7+, Debian 9+
- **Windows:** Limited support (use Docker or WSL2)

## Getting Help

**Diagnostic Tools:**

```bash
# Check package installation
python3 -c "import label_processing; print('✅ Package working')"

# Test system dependencies
tesseract --version
python3 -c "import cv2; print('✅ OpenCV working')"

# Run test suite
python3 -m pytest unit_tests/ -v
```

**Support Channels:**

- **GitHub Issues:** Bug reports and feature requests
- **Documentation:** Additional examples and tutorials
- **Community:** User discussions and tips

**Before Asking for Help:**

1. Try with the sample data first
2. Check that installation completed successfully
3. Run the diagnostic scripts
4. Include error messages and system information in reports
