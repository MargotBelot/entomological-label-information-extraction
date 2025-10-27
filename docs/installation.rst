Installation
============

This guide covers the installation process for the Entomological Label Information Extraction system.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~~

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.10 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB free space minimum
- **Conda**: Required for Python environment management
- **Tesseract OCR**: Required for text extraction
- **Docker**: Optional (for containerized execution or HPC)

Software Dependencies
~~~~~~~~~~~~~~~~~~~~~

Conda Installation
^^^^^^^^^^^^^^^^^^

Conda is **required** for managing the Python environment.

**All Platforms**

.. code-block:: bash

   # Download and install Miniconda
   # Visit: https://conda.io/miniconda.html
   
   # Verify installation
   conda --version

Tesseract OCR Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tesseract is **required** for optical character recognition.

**macOS**

.. code-block:: bash

   brew install tesseract
   
   # Verify installation
   tesseract --version

**Windows**

.. code-block:: bash

   # Download installer from:
   # https://github.com/UB-Mannheim/tesseract/wiki
   
   # After installation, verify:
   tesseract --version

**Linux (Ubuntu/Debian)**

.. code-block:: bash

   sudo apt update
   sudo apt install tesseract-ocr
   
   # Verify installation
   tesseract --version

Docker Installation (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Docker is **optional** - only needed for containerized execution or HPC environments.

**macOS**

.. code-block:: bash

   brew install --cask docker
   open /Applications/Docker.app

**Windows**

.. code-block:: bash

   # Download from: https://docker.com
   # Or: winget install Docker.DockerDesktop

**Linux**

.. code-block:: bash

   sudo apt install docker.io docker-compose
   sudo systemctl start docker
   sudo usermod -aG docker $USER  # Optional

Installation Methods
--------------------

Option 1: Conda Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-repo/entomological-label-information-extraction.git
   cd entomological-label-information-extraction

   # Create conda environment
   conda env create -f environment.yml

   # Activate environment
   conda activate entomological-label

   # Install package in development mode
   pip install -e .

Option 2: pip Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-repo/entomological-label-information-extraction.git
   cd entomological-label-information-extraction

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\\Scripts\\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Install package
   pip install -e .

Option 3: Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For developers who want to contribute:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-repo/entomological-label-information-extraction.git
   cd entomological-label-information-extraction

   # Create conda environment
   conda env create -f environment.yml
   conda activate entomological-label

   # Install with development dependencies
   pip install -e .[dev]

   # Install pre-commit hooks
   pre-commit install

Option 4: HPC/Cluster Installation (Apptainer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For high-performance computing environments:

.. code-block:: bash

   # Build Apptainer container
   cd pipelines
   apptainer build elie.sif elie.def
   
   # Transfer to HPC cluster
   scp elie.sif username@hpc.cluster.edu:/path/on/hpc/
   
   # Run on HPC
   apptainer run --bind /scratch/data:/app/data elie.sif mli

See ``pipelines/HPC_QUICKSTART.md`` for complete HPC documentation including SLURM job scripts.

Verification
------------

Test Installation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Verify conda environment
   conda activate entomological-label
   
   # Check that the package is installed
   python -c "import label_processing; print('✅ Installation successful!')"
   
   # Verify Tesseract is installed
   tesseract --version
   
   # Optional: Check Docker (if using containerized execution)
   docker --version

   # Run health check
   python scripts/health_check.py

Test Basic Functionality
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Launch the GUI to test the interface
   python launch_gui.py

   # Or test with sample data (if available)
   python scripts/processing/detection.py --help

Data Directory Setup
~~~~~~~~~~~~~~~~~~~~~

The system expects specific directory structures:

.. code-block:: bash

   # These directories should already exist in the repository
   ls data/MLI/input    # Multi-label input directory
   ls data/MLI/output   # Multi-label output directory
   ls data/SLI/input    # Single-label input directory
   ls data/SLI/output   # Single-label output directory

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Conda not found**
   Install Miniconda from https://conda.io/miniconda.html and restart your terminal.

**Tesseract not found**
   Install Tesseract: ``brew install tesseract`` (macOS) or ``sudo apt install tesseract-ocr`` (Linux).

**Docker not found** (optional)
   Only needed for containerized execution. Install from https://docker.com if needed.

**Permission denied with Docker (Linux)**
   Add your user to the docker group: ``sudo usermod -aG docker $USER`` and log out/in.

**Conda environment creation fails**
   Try updating conda: ``conda update conda`` and retry.

**Import errors**
   Make sure you've activated the environment: ``conda activate entomological-label``.

**Memory errors**
   Ensure you have sufficient RAM available. Close other applications if needed.

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the :doc:`troubleshooting` guide
2. Review the error messages carefully
3. Check system requirements are met
4. Consult the GitHub issues page
5. Contact the maintainers

Next Steps
----------

After successful installation:

1. Read the :doc:`quickstart` guide
2. Review the :doc:`user_guide`
3. Check the :doc:`api/modules` documentation
4. Try processing some sample images