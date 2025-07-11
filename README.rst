==============================================================
Collection Mining – Entomological Label Information Extraction
==============================================================

.. contents::
   :local:

Overview
========

This package provides a modular framework for the **semi-automated processing of entomological specimen labels**.  
It uses artificial intelligence to perform **label detection, classification, rotation correction, OCR, and clustering**, laying the groundwork for comprehensive information extraction.  
It is designed to work in conjunction with the `python-mfnb` package for downstream clustering tasks.

Key Features
============

- **AI-Powered Label Classification**: Three TensorFlow-based classifiers tailored to different label types.
- **OCR Pipeline**: Supports both Tesseract and the Google Cloud Vision API.
- **Modular Components**: For classification, preprocessing, text extraction, and postprocessing.
- **High Efficiency**: Optimized for digitizing large-scale entomological collections.

**Model Retraining Notebooks Provided**

We provide Jupyter notebooks in the ``training_notebooks/`` folder to allow users to retrain the models on their own data.  
These notebooks cover label detection, classification, and rotation correction, and can be adapted to new datasets as needed.

Prerequisites
=============

- Python 3.10 (for local installation)
- Docker Desktop (for running the pipeline in containers)
- Docker Compose
- (Optional) Conda for environment management
- (Optional) NVIDIA GPU and drivers for faster deep learning inference

Installation
============

1. Create a Python 3.10 environment (recommended to ensure dependency compatibility):

   .. code-block:: console

      conda create --name ELIE python=3.10

2. Activate the environment:

   .. code-block:: console

      conda activate ELIE

3. Install the package:

   .. code-block:: console

      cd entomological-label-information-extraction
      pip install .

4. Install Tesseract OCR (optional, required if using Tesseract):

   - **Ubuntu/Debian**:

     .. code-block:: console

        sudo apt install tesseract-ocr

   - **macOS**:

     .. code-block:: console

        brew install tesseract

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

   python scripts/processing/vision.py -d <path_to_cropped_images> -c <path_to_credentials.json> -o <output_directory>

Replace placeholders with your actual paths.

Installing `zbar` for QR Code Recognition
=========================================

To enhance QR code detection using `zbar`, install the following dependencies:

- **macOS**:

  .. code-block:: console

     brew install zbar

- **Linux**:

  .. code-block:: console

     sudo apt-get install libzbar0

- **Windows**: `zbar` is bundled with the Python wheels and requires no extra setup.

Docker Usage
============

Docker Installation
-------------------

Download and install Docker Desktop:

- https://www.docker.com/products/docker-desktop/

Verify Docker is installed:

.. code-block:: console

   docker --version

Docker Compose Installation
---------------------------

(Optional) Install Docker Compose via conda:

.. code-block:: console

   conda install -c conda-forge docker-compose

Verify Docker Compose:

.. code-block:: console

   docker-compose --version

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

Hardware Requirements
=====================

- Recommended: **NVIDIA GPU** for fast inference
- CPU-only systems are supported but significantly slower
- To enable GPU support in Docker:

  1. Install the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_
  2. Run Docker with GPU support:

     .. code-block:: console

        docker compose --gpus all -f multi-label-docker-compose.yaml up --build
