==============================================================
Collection Mining – Entomological Label Information Extraction
==============================================================

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :alt: Python Version
   :target: https://python.org

.. image:: https://img.shields.io/badge/license-GPL--3.0-green.svg
   :alt: License
   :target: LICENSE

.. contents::
   :local:
   :depth: 2

Overview
========

This package provides a **modular AI framework** for the semi-automated processing of entomological specimen labels. 
It combines state-of-the-art machine learning techniques to create a complete digitization pipeline that transforms 
physical specimen labels into structured, searchable data.

**Pipeline Workflow:**

.. image:: docs/images/pipeline_flowchart.png
   :alt: ELIE Pipeline Flowchart
   :align: center
   :width: 100%

**Legend:**

- **🔍 Detection Stage**: Automatically locate and extract individual labels from specimen images
- **🏷️ Classification Stage**: Categorize labels by characteristics (handwritten vs. printed, etc.)
- **🔄 Rotation Stage**: Correct text orientation for optimal OCR performance
- **📝 OCR Stage**: Extract text using Tesseract or Google Cloud Vision API
- **⚙️ Post-processing**: Clean, structure, and validate extracted information
- **📊 Output**: Generate CSV, JSON, and structured data files

**Process Flow:**

1. **Label Detection** → Automatically locate labels in specimen images
2. **Image Classification** → Categorize labels by type (handwritten, printed, etc.)
3. **Rotation Correction** → Align text for optimal OCR performance  
4. **Text Extraction** → Convert images to text using OCR technologies
5. **Post-processing** → Clean and structure extracted information

Key Features
============

🤖 **AI-Powered Processing**
  - Three specialized TensorFlow classifiers for different label types
  - Deep learning models for label detection and rotation correction
  - Optimized for entomological specimen workflows

📝 **Flexible OCR Pipeline**
  - **Tesseract OCR**: Free, offline text recognition
  - **Google Cloud Vision API**: Premium cloud-based OCR with superior accuracy
  - QR code detection and processing capabilities

🔧 **Modular Architecture**
  - Independent components for each processing stage
  - Docker containerization for easy deployment
  - Configurable pipelines for different use cases

⚡ **Production Ready**
  - Optimized for large-scale collections (thousands of specimens)
  - GPU acceleration support for faster inference
  - Comprehensive error handling and logging

📚 **Extensible & Reproducible**
  - Jupyter notebooks for model retraining on custom datasets
  - Well-documented APIs for integration with existing workflows
  - Complete unit test coverage

Datasets
========

The training and testing datasets used for model development are publicly available on Zenodo:  
`https://doi.org/10.7479/khac-x956 <https://doi.org/10.7479/khac-x956>`_

Installation Guide
==================

🚀 **Quick Setup for Mac and Linux**

This guide will get you up and running with the entomological label extraction pipeline in just a few minutes.

**Option 1: Docker Installation (Recommended for beginners)**

.. code-block:: console

   # 1. Install Docker Desktop
   # Mac: Download from https://www.docker.com/products/docker-desktop/
   # Linux: Follow instructions at https://docs.docker.com/desktop/install/linux-install/
   
   # 2. Clone the repository
   git clone https://github.com/[your-username]/entomological-label-information-extraction.git
   cd entomological-label-information-extraction
   
   # 3. Place your specimen images in the input folder
   mkdir -p data/MLI/input
   # Copy your .jpg images to data/MLI/input/
   
   # 4. Run the complete pipeline
   docker compose -f multi-label-docker-compose.yaml up --build
   
   # 5. Find results in data/MLI/output/

**Option 2: Python Installation (For developers)**

*Prerequisites:*

- Python 3.9+ (3.10 recommended)
- Git
- Package manager (pip/conda)

**Step 1: Clone the repository**

.. code-block:: console

   git clone https://github.com/[your-username]/entomological-label-information-extraction.git
   cd entomological-label-information-extraction

**Step 2: Choose your installation method**

**Method A: Conda (Recommended)**

.. code-block:: console

   # Install conda if you don't have it:
   # Mac: Download from https://docs.anaconda.com/anaconda/install/mac-os/
   # Linux: Download from https://docs.anaconda.com/anaconda/install/linux/
   
   # Create and activate environment
   conda env create -f environment.yml
   conda activate entomological-label
   
   # Install the package
   pip install -e .

**Method B: pip + venv**

.. code-block:: console

   # Create virtual environment
   python3 -m venv elie-env
   
   # Activate environment
   # Mac/Linux:
   source elie-env/bin/activate
   
   # Install the package
   pip install -e .

**Step 3: Install system dependencies**

**For macOS:**

.. code-block:: console

   # Install Homebrew if you don't have it
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install Tesseract OCR (for text recognition)
   brew install tesseract
   
   # Install zbar (for QR code detection)
   brew install zbar

**For Linux (Ubuntu/Debian):**

.. code-block:: console

   # Update package list
   sudo apt update
   
   # Install Tesseract OCR
   sudo apt install tesseract-ocr
   
   # Install zbar for QR codes
   sudo apt install libzbar0
   
   # Install additional dependencies
   sudo apt install python3-dev python3-pip

**For Linux (CentOS/RHEL/Fedora):**

.. code-block:: console

   # For CentOS/RHEL
   sudo yum install tesseract zbar-devel python3-devel
   
   # For Fedora
   sudo dnf install tesseract zbar-devel python3-devel

**Step 4: Verify installation**

.. code-block:: console

   # Test that everything is working
   python3 -c "import label_processing, label_postprocessing, label_evaluation, pipelines; print('✅ Installation successful!')"
   
   # Run tests to ensure everything works
   python3 -m pytest unit_tests/ -v

**Quick Start Guide**

**1. Basic Label Processing:**

.. code-block:: console

   # Process a folder of specimen images
   python3 scripts/processing/detection.py -j /path/to/your/images -o /path/to/output

**2. Custom Pipeline:**

.. code-block:: python

   from label_processing.label_detection import PredictLabel
   from label_processing.text_recognition import ocr_tesseract
   
   # Initialize the label detector
   detector = PredictLabel(
       path_to_model="models/detection_model.pth",
       classes=["label"]
   )
   
   # Process your images
   results = detector.class_prediction("/path/to/image.jpg")
   print(results)

**Troubleshooting**

**Common Issues:**

🔧 **"No module named 'label_processing'"**
   
   - Make sure you installed with ``pip install -e .`` from the project directory
   - Check that your virtual environment is activated

🔧 **"TesseractNotFoundError"**
   
   - Install Tesseract: ``brew install tesseract`` (Mac) or ``sudo apt install tesseract-ocr`` (Linux)
   - Add Tesseract to your PATH

🔧 **GPU/CUDA Issues**
   
   - The package works with CPU by default
   - For GPU acceleration, install PyTorch with CUDA support

🔧 **Permission Errors**
   
   - Use ``sudo`` for system-wide installations (not recommended)
   - Or use virtual environments (recommended)

🔧 **Model Loading Issues ("invalid load key" errors)**
   
   - Run the diagnostic script: ``python3 scripts/troubleshooting/test_model_loading.py``
   - This will test your model file and suggest fixes
   - Common cause: CUDA/CPU mismatch (model trained on GPU, loading on CPU)
   - The improved code automatically handles this issue

**Getting Help:**

- 📖 Check the `documentation <docs/>`_
- 🐛 Report issues on `GitHub Issues <https://github.com/[your-username]/entomological-label-information-extraction/issues>`_
- 💬 Ask questions in `Discussions <https://github.com/[your-username]/entomological-label-information-extraction/discussions>`_

**Next Steps:**

- See `Usage Examples <#usage-examples>`_ for detailed workflows
- Check out `Training Notebooks <training_notebooks/>`_ to customize models
- Read about `Docker Usage <#docker-usage>`_ for production deployment

Input Image Guidelines
======================

The modules work best on **JPEG** images that adhere to standardized practices, such as those from:

- `AntWeb <https://www.antweb.org/>`_
- `Bees & Bytes <https://www.zooniverse.org/projects/mfnberlin/bees-and-bytes>`_
- `Atlas of Living Australia <https://www.ala.org.au/>`_

Recommended image specifications:

- High-resolution JPEG format (300 DPI)
- Clear separation between labels
- Horizontal text alignment
- No insects or other elements in the image
- Consistent label positioning across images
- Preferably black background (white is acceptable)

Google Cloud Vision Setup
=========================

To use the Google Vision API:

1. Create a Google Cloud account.
2. Follow the setup instructions here:  
   `Google Vision API setup <https://cloud.google.com/vision/docs/setup>`_
3. Generate and download a **credentials JSON** file.

Run the OCR script independently:

.. code-block:: console

   python3 scripts/processing/vision.py -d <path_to_cropped_images> -c <path_to_credentials.json> -o <output_directory>

Replace placeholders with your actual paths.

Docker Usage
============

Docker Installation
-------------------

- Download and install Docker Desktop: https://www.docker.com/products/docker-desktop/
- Verify Docker is installed:

  .. code-block:: console

     docker --version

Pipeline Execution
------------------

This repository includes Dockerfiles and Docker Compose configurations.

**Available Compose Modes**:

- **Multi-label**: Full pipeline including label detection.
- **Single-label**: Pipeline without detection (e.g., cropped labels).

.. note::

   Example datasets for both pipelines are available in the ``data/`` folder.

**Run Multi-label Pipeline** (recommended):

.. code-block:: console

   docker compose -f multi-label-docker-compose.yaml up --build

This will:

1. Build all Docker images
2. Run detection, classification, OCR, and postprocessing

**Run Single-label Pipeline**:

.. code-block:: console

   docker compose -f single-label-docker-compose.yaml up --build

Final output will be saved in:

- ``data/MLI/`` for multi-label
- ``data/SLI/`` for single-label

To stop the pipeline at any time:

.. code-block:: console

   Ctrl+C

Troubleshooting
---------------

- **Docker must be running**: Ensure Docker Desktop is active.
- **Out-of-memory errors**: Increase memory allocation in Docker Desktop → Settings → Resources → Memory (8GB+ recommended).
- **Missing files**: Ensure images are placed in the correct ``data/`` subfolders.
- **Build changes**: Use ``--build`` when modifying Dockerfiles.
- **Missing libraries**: Ensure required dependencies (e.g., ``cv2``, ``libGL.so.1``) are installed.
- **Orphan containers**:

  .. code-block:: console

     docker compose -f multi-label-docker-compose.yaml down --remove-orphans

Usage Examples
==============

**Processing Individual Components:**

.. code-block:: python

   from label_processing.label_detection import PredictLabel
   from label_processing.tensorflow_classifier import class_prediction
   from label_processing.text_recognition import ImageProcessor, Tesseract
   
   # Label detection
   detector = PredictLabel(model_path, ["label"], image_path)
   predictions = detector.class_prediction(image_path)
   
   # Classification
   model = get_model(classifier_path)
   df = class_prediction(model, ["handwritten", "printed"], image_dir)
   
   # OCR processing
   processor = ImageProcessor.read_image(image_path)
   ocr = Tesseract(image=processor)
   text_result = ocr.image_to_string()

**Command Line Usage:**

.. code-block:: console

   # Run label detection
   python3 scripts/processing/detection.py \
     --input data/input/ \
     --output data/detection/ \
     --model models/label_detection_model.pth
   
   # Run classification
   python3 scripts/processing/classifiers.py \
     --input data/detection/ \
     --output data/classified/ \
     --model models/label_classifier_hp/
   
   # Run OCR with Tesseract
   python3 scripts/processing/tesseract.py \
     --input data/classified/ \
     --output data/ocr_results/

Output Format
=============

**Current Output:**

The pipeline currently generates output in the following formats:

**CSV Files:**
  - Detection predictions with bounding box coordinates
  - Classification results for label types
  - OCR text extraction results

**Image Files:**
  - Cropped label images organized by classification
  - Processed images after rotation correction

**Planned Enhancements:**

.. note::
   The following features are planned for future releases:

- **Unified JSON Output**: Complete structured data combining all pipeline stages
- **Metadata Files**: Processing parameters and pipeline statistics
- **Structured Text Fields**: Automated parsing of taxonomic information and dates
- **Quality Metrics**: Confidence scores and validation indicators

**Example Future Output Structure:**

.. code-block:: json

   {
     "filename": "specimen_001.jpg",
     "labels": [
       {
         "bbox": [100, 150, 300, 250],
         "classification": "printed",
         "confidence": 0.95,
         "text": "Lepidoptera\nNoctuidae\nCollected: 1995-07-15",
         "qr_code": null,
         "processed_text": {
           "family": "Noctuidae",
           "order": "Lepidoptera",
           "collection_date": "1995-07-15"
         }
       }
     ]
   }

Development & Testing
====================

**Running Tests:**

.. code-block:: console

   # Run all tests
   python3 -m pytest unit_tests/ -v
   
   # Run with coverage
   pip install pytest pytest-cov
   python3 -m pytest unit_tests/ --cov=. --cov-report=html
   
   # Run specific test modules
   python3 -m pytest unit_tests/label_processing_tests/

**Code Quality:**

.. code-block:: console

   # Install development dependencies
   pip install -e .[dev]
   
   # Run code formatting
   black .
   isort .
   
   # Run linting
   flake8 .
   mypy .
   
   # Set up pre-commit hooks
   pre-commit install

Model Retraining
================

Customize the models for your specific datasets using the provided Jupyter notebooks:

**Available Training Notebooks:**

- ``training_notebooks/Label_Detection_Detecto_Training_Notebook.ipynb``
  - Retrain the label detection model on custom specimen images
  - Supports custom annotation formats and label types

- ``training_notebooks/Classifier_TensorFlow_Training_Notebook.ipynb``
  - Train classification models for different label characteristics
  - Includes data augmentation and transfer learning techniques

- ``training_notebooks/Label_Rotation_TensorFlow_Training_Notebook.ipynb``
  - Develop rotation correction models for your image types
  - Handles various rotation angles and image qualities

**Training Data Requirements:**

- **Detection**: Annotated images with bounding box coordinates
- **Classification**: Labeled image crops organized by category
- **Rotation**: Image pairs (original and corrected orientations)

Hardware Requirements
====================

**Minimum Requirements:**
- **CPU**: 4+ cores
- **RAM**: 8GB+ (16GB+ recommended for large datasets)
- **Storage**: 5GB+ free space
- **OS**: macOS 10.14+, Ubuntu 18.04+, or other Linux distribution

**Recommended for Production:**
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 32GB+
- **Storage**: SSD with 50GB+ free space

**GPU Support:**

To enable GPU acceleration:

1. Install NVIDIA drivers and CUDA toolkit
2. Install PyTorch with CUDA support:

   .. code-block:: console

      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

3. For Docker GPU support, install `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_

License
=======

This project is licensed under the GPL-3.0 License - see the `LICENSE <LICENSE>`_ file for details.

Citation
========

If you use this software in your research, please cite the associated dataset:

.. code-block:: bibtex

   @dataset{entomological_labels_2024,
     title={Entomological Label Information Extraction Dataset},
     url={https://doi.org/10.7479/khac-x956},
     DOI={10.7479/khac-x956},
     publisher={Zenodo},
     year={2025}
   }

Contributing
============

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

**Development Setup:**

.. code-block:: console

   # Clone your fork
   git clone https://github.com/your-username/entomological-label-information-extraction.git
   cd entomological-label-information-extraction
   
   # Install in development mode
   pip install -e .[dev]
   
   # Set up pre-commit hooks
   pre-commit install
   
   # Run tests
   python3 -m pytest unit_tests/ -v
