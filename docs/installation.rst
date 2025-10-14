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
- **Docker**: Required for pipeline processing

Software Dependencies
~~~~~~~~~~~~~~~~~~~~~

Docker Installation
^^^^^^^^^^^^^^^^^^^

Docker is **required** for running the processing pipelines.

**macOS**

.. code-block:: bash

   # Download and install Docker Desktop
   # Visit: https://desktop.docker.com/mac/main/amd64/Docker.dmg (Intel)
   # Visit: https://desktop.docker.com/mac/main/arm64/Docker.dmg (Apple Silicon)

   # Or install via Homebrew
   brew install --cask docker

   # Start Docker Desktop
   open /Applications/Docker.app

**Windows**

.. code-block:: powershell

   # Download and install Docker Desktop
   # Visit: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe

   # Or install via Chocolatey
   choco install docker-desktop

   # Or install via winget
   winget install Docker.DockerDesktop

**Linux (Ubuntu/Debian)**

.. code-block:: bash

   # Update package index
   sudo apt update

   # Install Docker
   sudo apt install docker.io docker-compose

   # Start and enable Docker
   sudo systemctl start docker
   sudo systemctl enable docker

   # Add user to docker group (optional, avoids sudo)
   sudo usermod -aG docker $USER

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

Verification
------------

Test Installation
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check that the package is installed
   python -c "import label_processing; print('Installation successful!')"

   # Check Docker is working
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

**Docker not found**
   Make sure Docker is installed and running. On Windows/macOS, start Docker Desktop.

**Permission denied (Linux)**
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