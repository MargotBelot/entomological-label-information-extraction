Collection Mining – Entomological Label Information Extraction
==============================================================

*A Python package developed at the Berlin Natural History Museum*

.. contents::

Overview
--------

This package provides a modular framework for the **semi-automated processing of entomological specimen labels**. 
It uses artificial intelligence to perform **label detection, classification, rotation correction, OCR, and clustering** laying the groundwork for comprehensive information extraction. 
It is designed to work in conjunction with the 'python-mfnb' cpackage for downstream clustering tasks.

Key Features
------------

- **AI-Powered Label Classification**: Three TensorFlow-based classifiers tailored to different label types.
- **OCR Pipeline**: Supports both Tesseract and the Google Cloud Vision API.
- **Modular Components**: For classification, preprocessing, text extraction, and postprocessing.
- **High Efficiency**: Optimized for digitizing large-scale entomological collections.


Prerequisites
-------------

- Python 3.10 (for local installation)
- Docker Desktop (for running the pipeline in containers)
- Docker Compose
- (Optional) Conda for environment management
- (Optional) NVIDIA GPU and drivers for faster deep learning inference


Installation
------------

1. Create a Python 3.10 environment (recommended to ensure dependency compatibility):

.. code-block:: console

  conda create --name ELIE python=3.10

Be sure to activate your environment before running any further commands:

.. code-block:: console

  conda activate ELIE

2. Install the package:

.. code-block:: console

  cd entomological-label-information-extraction
  pip install .

3. Install Tesseract (optional, required if using Tesseract OCR):

- **Ubuntu/Debian**:

.. code-block:: console

  sudo apt install tesseract-ocr

- **macOS**:

.. code-block:: console

  brew install tesseract


Input Image Guidelines
----------------------

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
--------------------------

To use the Google Vision API:

1. Create a Google Cloud account.
2. Follow the setup instructions here: `Google Vision API setup <https://cloud.google.com/vision/docs/setup>`_.
3. Generate and download a **credentials JSON** file.

You can run the OCR script independently by providing your credentials file:

.. code-block:: console

   python scripts/processing/vision.py -d <path_to_cropped_images> -c <path_to_credentials.json> -o <output_directory>

Replace ``<path_to_cropped_images>``, ``<path_to_credentials.json>``, and ``<output_directory>`` with your actual paths.

The script will use your Google Cloud Vision credentials to process all images in the specified directory and save the results in the output directory.


Installing `zbar` for QR Code Recognition
-----------------------------------------

To enhance QR code detection using `zbar`, install the following dependencies:

- **macOS**:

.. code-block:: console

  brew install zbar

- **Linux**:

.. code-block:: console

  sudo apt-get install libzbar0

On Windows, zbar is already bundled with the Python binaries.


Docker Usage
============

Docker Installation
-------------------

Docker is required to run the pipeline. Download and install Docker Desktop from:

- https://www.docker.com/products/docker-desktop/

After installation, restart your terminal and verify Docker is installed:

.. code-block:: console

  docker --version


Docker Compose Installation
---------------------------

If you are using conda and want to install Docker Compose in your environment, run:

.. code-block:: console

  conda install -c conda-forge docker-compose

After installation, verify Docker Compose is available:

.. code-block:: console

  docker-compose --version


Pipeline Execution
------------------

This repository includes Dockerfiles for each processing module, as well as a Docker Compose setup to orchestrate them.

**Available Compose Modes**:

- **Multi-label**: Full pipeline including label detection.
- **Single-label**: Runs the pipeline without label detection.

**Before you start:**

- **Make sure Docker Desktop is running.**  
  You must start Docker Desktop before running any Docker or Docker Compose commands.
- **(Recommended) Increase Docker’s memory allocation:**  
  For best performance, especially when running the detection model, open Docker Desktop → Settings → Resources and set the memory to at least **4GB** (preferably 8GB+).

**Usage:**

From the root directory, run:

.. note::

   Example datasets for both Single-label (SLI) and Multi-label (MLI) pipelines are already included in the ``data`` folder. You can use these to immediately test the Docker Compose pipelines without any additional setup.

**Multi-label pipeline (recommended):**

.. code-block:: console

  docker compose -f multi-label-docker-compose.yaml up --build

This command will:
  1. Build all required Docker images.
  2. Run the full pipeline, including detection, classification, OCR, and postprocessing.

**Single-label pipeline:**

.. code-block:: console

  docker compose -f single-label-docker-compose.yaml up --build

After the pipeline completes, the final output files can be found in the ``data/SLI/`` (Single-label) or ``data/MLI/`` (Multi-label) directory in the project folder.

To stop the pipeline at any time, press ``Ctrl+C`` in your terminal.

Troubleshooting
---------------

- **Docker must be running:** If you see errors about Docker not being found, make sure Docker Desktop is started.
- **Increase memory if detection fails:** If the detection service fails with an "out of memory" or exit code 137, increase Docker’s memory allocation in Docker Desktop settings.
- If you see errors about missing files or directories, ensure your input images are placed in the correct ``data/`` subfolders as described above.
- If you change any code or Dockerfiles, always use the ``--build`` flag to rebuild images.
- For errors about missing Python packages or libraries (e.g., ``cv2`` or ``libGL.so.1``), make sure your requirements files and Dockerfiles are up to date.
- If you see a warning about orphan containers, you can remove them with:

.. code-block:: console

  docker compose -f multi-label-docker-compose.yaml down --remove-orphans


Hardware Requirements
---------------------

For optimal performance, especially when running deep learning models (e.g., label detection, rotation correction), it is recommended to use a machine with a dedicated NVIDIA GPU and recent drivers. While the pipeline can run on CPU-only systems, processing will be significantly slower.

If you plan to use GPU acceleration with Docker, ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed and configured.

If your system supports GPU acceleration and you have set up the NVIDIA Container Toolkit, you can run the pipeline with GPU support by adding the ``--gpus all`` flag:

.. code-block:: console

  docker compose --gpus all -f multi-label-docker-compose.yaml up --build
