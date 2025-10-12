# Installation Guide

This guide provides detailed instructions for installing and setting up the Entomological Label Information Extraction system.

## 📋 System Requirements

### Minimum Requirements

- **Operating System**: macOS 10.14+, Ubuntu 18.04+, Windows 10+
- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB recommended for large batches)
- **Storage**: 5GB free space (more for model storage and processing)
- **GPU**: Optional but recommended for faster processing

### Recommended Requirements

- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support (for accelerated processing)
- **Storage**: SSD with 20GB+ free space

## 🔧 Installation Methods

### Method 1: Conda (Recommended)

This is the recommended installation method as it handles all dependencies automatically.

#### Step 1: Install Conda

If you don't have conda installed:

**macOS:**
```bash
# Install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh
```

**Linux:**
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Windows:**
Download and run the installer from: https://conda.io/miniconda.html

#### Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/entomological-label-information-extraction.git
cd entomological-label-information-extraction

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate entomological-label

# Install the package in development mode
pip install -e .
```

#### Step 3: Verify Installation

```bash
# Run health check
python scripts/health_check.py

# Run a quick test
python -c "import label_processing; print('✓ Installation successful!')"
```

### Method 2: pip (Advanced Users)

If you prefer using pip or need a lighter installation:

#### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 2: Install Dependencies

```bash
# Clone repository
git clone https://github.com/your-username/entomological-label-information-extraction.git
cd entomological-label-information-extraction

# Install the package
pip install -e .

# Install optional dependencies
pip install -e .[dev]  # For development
pip install -e .[docs] # For documentation
```

## 🐳 Docker Installation (Optional)

For isolated and reproducible deployments:

### Prerequisites

- Docker Desktop (macOS/Windows) or Docker Engine (Linux)
- Docker Compose

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/entomological-label-information-extraction.git
cd entomological-label-information-extraction

# Build Docker containers
docker-compose -f pipelines/multi-label-docker-compose.yaml build

# Run a test container
docker-compose -f pipelines/multi-label-docker-compose.yaml up
```

## 🔍 System-Specific Setup

### macOS Setup

#### Install System Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Tesseract OCR
brew install tesseract

# Install Git (if not already installed)
brew install git
```

#### M1/M2 Mac Considerations

For Apple Silicon Macs, ensure you're using native ARM64 packages:

```bash
# Use conda-forge channel for ARM64 packages
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### Linux Setup (Ubuntu/Debian)

#### Install System Dependencies

```bash
# Update package lists
sudo apt update

# Install Python and development tools
sudo apt install python3.10 python3.10-dev python3.10-venv
sudo apt install build-essential

# Install Tesseract OCR
sudo apt install tesseract-ocr tesseract-ocr-eng

# Install Git
sudo apt install git

# Install additional libraries for image processing
sudo apt install libgl1-mesa-glx libglib2.0-0
```

### Windows Setup

#### Using Windows Subsystem for Linux (Recommended)

```bash
# Install WSL2
wsl --install

# Inside WSL2, follow the Linux setup instructions above
```

#### Native Windows Installation

```powershell
# Install Python 3.10 from Microsoft Store or python.org

# Install Tesseract OCR
# Download from: https://github.com/tesseract-ocr/tesseract/releases
# Add to PATH after installation

# Install Git
# Download from: https://git-scm.com/download/win
```

## 🧪 Verification

After installation, run these commands to verify everything is working:

### Basic Verification

```bash
# Activate environment (if using conda)
conda activate entomological-label

# Run comprehensive health check
python scripts/health_check.py

# Test core imports
python -c "
import tensorflow as tf
import torch
import cv2
import label_processing
import label_evaluation
print('✓ All core modules imported successfully')
print(f'TensorFlow: {tf.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
"
```

### Run Test Suite

```bash
# Run all tests
pytest unit_tests/ -v

# Run quick smoke tests only
pytest unit_tests/ -m "not slow" -v
```

### GUI Test

```bash
# Launch GUI (should open without errors)
python launch_gui.py
```

## 🔧 Troubleshooting

### Common Issues

#### ImportError: No module named 'cv2'

**Solution:**
```bash
pip install opencv-python==4.9.0.80
```

#### Tesseract not found

**macOS:**
```bash
brew install tesseract
export PATH="/opt/homebrew/bin:$PATH"  # Add to ~/.bashrc or ~/.zshrc
```

**Linux:**
```bash
sudo apt install tesseract-ocr
```

**Windows:**
Download from GitHub releases and add to PATH.

#### CUDA Issues (GPU Support)

If you have an NVIDIA GPU but CUDA is not detected:

```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-compatible PyTorch
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch
```

#### Permission Errors on macOS

```bash
# If you get permission errors, try:
sudo xcode-select --install
```

#### Memory Issues

If you encounter out-of-memory errors:

1. Reduce batch sizes in configuration
2. Close other applications
3. Consider using Docker with memory limits

### Environment Issues

#### Conda Environment Conflicts

```bash
# Clean conda environment
conda clean --all

# Remove and recreate environment
conda env remove -n entomological-label
conda env create -f environment.yml
```

#### Python Version Issues

Ensure you're using Python 3.10+:

```bash
python --version
# Should show Python 3.10.x or higher
```

If not, install the correct version:

```bash
# macOS with Homebrew
brew install python@3.10

# Ubuntu/Debian
sudo apt install python3.10
```

## 🚀 Performance Optimization

### GPU Acceleration

For faster processing, enable GPU support:

1. **Install CUDA** (NVIDIA GPUs only)
2. **Install GPU-optimized packages:**
   ```bash
   # For TensorFlow
   pip install tensorflow-gpu==2.15.0
   
   # For PyTorch
   conda install pytorch torchvision cudatoolkit -c pytorch
   ```

### Memory Optimization

For large datasets:

1. **Increase system virtual memory**
2. **Use batch processing**
3. **Configure memory limits in Docker:**
   ```yaml
   services:
     processing:
       deploy:
         resources:
           limits:
             memory: 8G
   ```

## 📦 Development Installation

For contributors and developers:

```bash
# Clone with development branch
git clone -b develop https://github.com/your-username/entomological-label-information-extraction.git
cd entomological-label-information-extraction

# Install in development mode with all dependencies
conda env create -f environment.yml
conda activate entomological-label
pip install -e .[dev,test,docs]

# Set up pre-commit hooks
pre-commit install

# Verify development setup
pytest unit_tests/ --cov=label_processing --cov-report=html
```

## 🆘 Getting Help

If you encounter issues during installation:

1. **Check the [GitHub Issues](https://github.com/your-username/entomological-label-information-extraction/issues)**
2. **Run the health check script:** `python scripts/health_check.py`
3. **Check system requirements** and ensure all dependencies are installed
4. **Create a new issue** with your system information and error messages

### System Information for Bug Reports

When reporting issues, include this information:

```bash
# Get system info
python scripts/health_check.py > system_info.txt

# Include Python environment info
pip list > pip_packages.txt
conda list > conda_packages.txt  # if using conda
```

---

**Next Steps:** After successful installation, see the [README.md](README.md) for usage instructions and examples.