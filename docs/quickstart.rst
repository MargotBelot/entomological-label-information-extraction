Quick Start Guide
=================

🎆 **Get running in 5 minutes!** Perfect for first-time users who want to test the system quickly.

.. tip::
   📋 **Prerequisites**: Make sure you have `Git <https://git-scm.com/>`_ and `Conda <https://conda.io/miniconda.html>`_ installed first!

🚀 Option 1: Gemini Pipeline (Recommended)
--------------------------------------------

**The simplest and most powerful way to get started — handles printed AND handwritten labels:**

.. code-block:: bash

   # 1. Get the code
   git clone https://github.com/MargotBelot/entomological-label-information-extraction.git
   cd entomological-label-information-extraction

   # 2. One-command setup
   conda env create -f environment.yml
   conda activate ELIE
   pip install -e .

   # 3. Set up your Gemini API key (free from https://aistudio.google.com/apikey)
   export GEMINI_API_KEY=<your-api-key>

   # 4. Add your images
   cp /path/to/your/photos/*.jpg data/MLI/input/

   # 5. Launch the interface
   python launch.py

**That's it!** 🎉 The Streamlit web interface will open where you can:

- Select the Gemini pipeline and configure options
- Monitor real-time processing progress
- Browse results and correct OCR text directly
- View extracted entities (species, collectors, dates, localities)
- Re-run entity recognition after corrections
- Export Darwin Core records (JSON/CSV)

🔧 Option 2: Traditional Pipelines (Offline)
---------------------------------------------

**For offline use with local models (Detectron2 + TensorFlow + Tesseract). Printed labels only.**

.. code-block:: bash

   # 1. Get the code
   git clone https://github.com/MargotBelot/entomological-label-information-extraction.git
   cd entomological-label-information-extraction

   # 2. Setup environment
   conda env create -f environment.yml
   conda activate ELIE
   pip install -e .

   # 3. Install Tesseract OCR (required for traditional pipelines)
   # macOS: brew install tesseract
   # Linux: sudo apt install tesseract-ocr

   # 4. Choose your interface:
   # Streamlit (recommended)
   python launch.py
   
   # OR Desktop GUI (Tkinter-based)
   python interfaces/launch_gui.py
   
   # OR Manual pipeline scripts
   ./tools/pipelines/run_gemini_pipeline_conda.sh  # Gemini (recommended)
   ./tools/pipelines/run_mli_pipeline_conda.sh     # Multi-label (traditional)
   ./tools/pipelines/run_sli_pipeline_conda.sh     # Single-label (traditional)

🎯 What Happens Next?
----------------------

After processing, you'll find your results in the output folders:

**Gemini Pipeline Output:**

.. code-block:: text

   data/MLI/output/
   ├── entity_master.json          # 📊 All labels with entities, GBIF, OSM
   ├── consolidated_results.json   # 📝 Labels with OCR text and metadata
   ├── quality_report.json         # ✅ Extraction quality scores
   ├── darwin_core.json            # 🧬 Darwin Core formatted records
   ├── darwin_core.csv             # 📄 Same in CSV format
   └── input_cropped/              # 🖼️ Cropped label images

**Traditional Pipeline Output:**

.. code-block:: text

   data/MLI/output/
   ├── consolidated_results.json    # 📊 Complete summary
   ├── input_predictions.csv       # 🗺 Label locations
   └── input_cropped/              # 🖼️ Cropped label images

   data/SLI/output/
   ├── consolidated_results.json    # 📊 Complete summary
   ├── corrected_transcripts.json  # 🧹 Clean text results
   └── classification/             # 📁 Sorted by label type

📈 Quick Results Check
-----------------------

Open ``consolidated_results.json`` to see all your extracted text and confidence scores!

.. code-block:: bash

   # Preview your results
   cat data/SLI/output/consolidated_results.json | head -20

🚑 Need Help?
---------------

- **Weird results?** → Check :doc:`troubleshooting`
- **Ready for production?** → Read the full :doc:`user_guide`
- **Want to contribute?** → See :doc:`contributing`
- **Found a bug?** → Report it on `GitHub Issues <https://github.com/MargotBelot/entomological-label-information-extraction/issues>`_

Understanding Pipeline Types
----------------------------

Multi-Label Images (MLI)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Use when**: You have full specimen photos containing multiple labels

.. code-block:: bash

   # Place images here
   data/MLI/input/specimen_001.jpg
   data/MLI/input/specimen_002.jpg

**What happens**:
1. System detects individual labels in each image
2. Crops each detected label
3. Saves cropped labels for further processing
4. Generates detection results

**Output**: Detected labels and bounding box coordinates

Single-Label Images (SLI)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use when**: You have pre-cropped individual label images

.. code-block:: bash

   # Place images here
   data/SLI/input/label_001.jpg
   data/SLI/input/label_002.jpg

**What happens**:
1. Classifies each label (empty/handwritten/printed/identifier)
2. Corrects rotation if needed
3. Extracts text using OCR
4. Post-processes and structures results

**Output**: Structured text data with metadata

Basic Usage Examples
--------------------

Streamlit Interface (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Quick launch
   python launch.py
   
   # OR launch Streamlit directly
   streamlit run interfaces/launch_streamlit.py

The Streamlit interface provides:

- Interactive web-based UI with pipeline selection (Gemini, MLI, SLI)
- Real-time progress tracking with job duration display
- Live processing dashboard with system metrics
- Results browser with file preview
- OCR text correction and entity recognition re-run
- Darwin Core export

Command Line Method
~~~~~~~~~~~~~~~~~~~

**Gemini Pipeline (Recommended):**

.. code-block:: bash

   # Full Gemini pipeline — detection, classification, OCR, entities
   ./tools/pipelines/run_gemini_pipeline_conda.sh

   # With custom options
   INPUT_DIR=data/MLI/input OUTPUT_DIR=data/MLI/output \
   ENTITY_RECOGNITION=true EXPORT_DWC=true EXPORT_CSV=true \
   ./tools/pipelines/run_gemini_pipeline_conda.sh

**Traditional Multi-Label Processing:**

.. code-block:: bash

   # Run detection on multi-label images
   python scripts/processing/detection.py -j data/MLI/input -o data/MLI/output

**Traditional Single-Label Processing:**

.. code-block:: bash

   # Run SLI components sequentially
   python scripts/processing/analysis.py -i data/SLI/input -o data/SLI/output  # empty label filtering
   python scripts/processing/classifiers.py -m 1 -j data/SLI/input -o data/SLI/output  # identifier/not_identifier
   python scripts/processing/classifiers.py -m 2 -j data/SLI/input -o data/SLI/output  # handwritten/printed
   python scripts/processing/rotation.py -i data/SLI/output/printed -o data/SLI/output/printed/rotated
   
   # OCR (choose one)
   python scripts/processing/tesseract.py -d data/SLI/output/printed/rotated -o data/SLI/output
   python scripts/processing/vision.py -c credentials.json -d data/SLI/output/printed/rotated -o data/SLI/output

Pipeline Scripts
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Gemini pipeline (recommended)
   ./tools/pipelines/run_gemini_pipeline_conda.sh

   # Multi-label pipeline (traditional, conda-based)
   ./tools/pipelines/run_mli_pipeline_conda.sh

   # Single-label pipeline (traditional, conda-based)
   ./tools/pipelines/run_sli_pipeline_conda.sh

Understanding Results
---------------------

Multi-Label Results
~~~~~~~~~~~~~~~~~~~

After MLI processing, you'll find:

.. code-block:: text

   data/MLI/output/
   ├── input_predictions.csv          # Detection results
   ├── input_cropped/                 # Cropped label images
   │   ├── specimen_001_label_0.jpg
   │   ├── specimen_001_label_1.jpg
   │   └── ...
   └── consolidated_results.json      # Summary report

Single-Label Results
~~~~~~~~~~~~~~~~~~~~

After SLI processing, you'll find:

.. code-block:: text

   data/SLI/output/
   ├── empty/                         # Empty labels
   ├── handwritten/                   # Manual transcription needed
   ├── printed/                       # OCR processing
   │   └── rotated/                   # Rotation-corrected labels
   ├── identifier/                    # QR codes, barcodes
   ├── ocr_preprocessed.json          # Tesseract results
   ├── ocr_google_vision.json         # Google Vision results
   ├── corrected_transcripts.json     # Cleaned text
   ├── plausible_transcripts.json     # High-confidence text
   └── consolidated_results.json      # Final structured output

Key Output Files
~~~~~~~~~~~~~~~~

**consolidated_results.json**
   Complete results with all extracted text, confidence scores, and metadata

**corrected_transcripts.json**
   Post-processed text with corrections and standardizations

**plausible_transcripts.json**
   High-confidence extractions suitable for automated processing

Common Workflows
----------------

Museum Digitization
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Photograph specimens (multi-label images)
   # 2. Process with MLI pipeline
   python scripts/processing/detection.py -j photos/ -o detections/
   
   # 3. Move cropped labels to SLI input
   mv detections/input_cropped/* data/SLI/input/
   
   # 4. Process individual labels
   python scripts/processing/analysis.py -j data/SLI/input -o data/SLI/output

Research Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Process pre-cropped labels directly
   python scripts/processing/analysis.py -j research_labels/ -o results/
   
   # 2. Extract high-confidence text
   cat results/plausible_transcripts.json
   
   # 3. Run evaluation metrics
   python scripts/evaluation/ocr_eval.py -i results/

Quality Assessment
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate comprehensive evaluation report
   python scripts/evaluation/analysis_eval.py -i data/SLI/output/
   
   # Check clustering analysis
   python scripts/evaluation/cluster_eval.py -i data/SLI/output/
   
   # Evaluate classification accuracy
   python scripts/evaluation/classifiers_eval.py -i data/SLI/output/

Next Steps
----------

Now that you have the basics working:

1. **User Guide**: Read the :doc:`user_guide` for end‑to‑end instructions
2. **API Documentation**: Browse :doc:`api/modules` for programmatic usage
3. **Troubleshooting**: Consult :doc:`troubleshooting` for common issues
4. **Contributing**: See :doc:`contributing` to get involved

Tips for Success
----------------

Image Quality
~~~~~~~~~~~~~

- Use high-resolution images (300+ DPI)
- Ensure good lighting and contrast
- Minimize blur and skew

Batch Processing
~~~~~~~~~~~~~~~~

- Process images in batches of 10-50 for optimal performance
- Monitor memory usage with large datasets
- Use Docker for consistent results across systems

Result Validation
~~~~~~~~~~~~~~~~~

- Always review high-confidence results manually
- Check empty label classifications
- Verify handwritten label identification

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

- Use GPU acceleration when available (traditional pipelines)
- Adjust batch sizes based on available memory
- Consider the Gemini pipeline for best accuracy on both printed and handwritten labels
