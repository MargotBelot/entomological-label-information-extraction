User Guide
==========

This comprehensive guide covers all aspects of using the Entomological Label Information Extraction system.

System Overview
---------------

The system is designed to extract and digitize text from museum specimen labels using AI and OCR technologies. It supports two main processing pipelines:

- **Multi-Label Images (MLI)**: Full specimen photos with multiple labels
- **Single-Label Images (SLI)**: Pre-cropped individual label images

Architecture
~~~~~~~~~~~~

.. code-block:: text

   Input Images → Detection → Classification → OCR → Post-processing → Structured Output

Core Components:
- Label detection using Faster R-CNN
- Classification models for label types
- OCR using Tesseract and Google Vision API
- Post-processing for text cleaning and structuring

Preprocessing and Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Stage 1 (Image Processing) is restricted to geometric normalization and routing only: label detection and cropping, classification (identifier vs. not, handwritten vs. printed, multi- vs single‑label), and rotation normalization to 0°/90°/180°/270°. No intensity-based enhancements (e.g., CLAHE, histogram equalization, global normalization) are applied in Stage 1 to preserve cues learned by the detectors/classifiers.
- Stage 2 (OCR preprocessing, printed labels) applies grayscale conversion, Gaussian/median denoising, binarization via Otsu or adaptive mean/Gaussian (block size and C tunable), skew estimation within ±10° and deskew, and optional morphological clean-up (dilation/erosion) before Tesseract OCR. Google Vision is called on the rotated ROI without thresholding.
- Empty‑label detection thresholds: we crop a 10% border on all sides, count “dark” pixels as mean RGB < 100, and classify a label as empty if the dark‑pixel proportion p_dark < 0.01 (1%).

Preparing Your Data
-------------------

Image Requirements
~~~~~~~~~~~~~~~~~~

**Quality Guidelines:**
- Resolution: 300 DPI or higher recommended
- Format: JPEG, PNG
- Lighting: Even, sufficient contrast
- Focus: Sharp, minimal blur
- Orientation: Any (system handles rotation)

**Multi-Label Images:**
- Full specimen photos showing multiple labels
- Include collection labels, determination labels, locality labels
- Ensure all labels are visible and readable

**Single-Label Images:**
- Individual label images, pre-cropped
- One label per image
- Include some margin around the label text

Directory Structure
~~~~~~~~~~~~~~~~~~~

Organize your data as follows:

.. code-block:: text

   project/
   ├── data/
   │   ├── MLI/
   │   │   ├── input/          # Multi-label input images
   │   │   └── output/         # Processing results
   │   └── SLI/
   │       ├── input/          # Single-label input images
   │       └── output/         # Processing results

Using the Interface
-------------------

Starting the Interface
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Recommended: Quick launch
   python launch.py
   
   # Alternative: Streamlit directly
   streamlit run interfaces/launch_streamlit.py
   
   # Alternative: Desktop GUI
   python interfaces/launch_gui.py

The **Streamlit web interface** (recommended) provides:

- **Interactive Web UI**: Modern browser-based interface
- **Real-time Progress**: Live progress tracking with job duration display
- **Processing Dashboard**: System metrics and performance monitoring
- **Results Browser**: Interactive file preview and analysis
- **Docker Management**: Automatic Docker status checking and startup

Interface Workflow
~~~~~~~~~~~~~~~~~~

1. **Select Input Directory**: Browse and choose folder containing your images
2. **Choose Pipeline Type**: MLI for specimen photos, SLI for cropped labels
3. **Configure Settings**: Set batch size and processing options
4. **Start Processing**: Click "Start Processing" and monitor real-time progress
5. **View Results**: Browse generated files, charts, and structured data
6. **Track Performance**: See total job duration and processing metrics

Command Line Usage
------------------

Basic Commands
~~~~~~~~~~~~~~

**Multi-Label Processing:**

.. code-block:: bash

   # Basic detection
   python scripts/processing/detection.py -j data/MLI/input -o data/MLI/output

   # With custom confidence threshold
   python scripts/processing/detection.py -j data/MLI/input -o data/MLI/output --confidence 0.7

**Single-Label Processing (sequential):**

.. code-block:: bash

   # 1) Empty label filtering
   python scripts/processing/analysis.py -i data/SLI/input -o data/SLI/output

   # 2) Classify identifiers and text type
   python scripts/processing/classifiers.py -m 1 -j data/SLI/input -o data/SLI/output   # identifier/not_identifier
   python scripts/processing/classifiers.py -m 2 -j data/SLI/input -o data/SLI/output   # handwritten/printed

   # 3) Rotation correction for printed labels
   python scripts/processing/rotation.py -i data/SLI/output/printed -o data/SLI/output/printed/rotated

   # 4) OCR (choose one)
   # Tesseract
   python scripts/processing/tesseract.py -d data/SLI/output/printed/rotated -o data/SLI/output
   # Google Vision
   python scripts/processing/vision.py -c credentials.json -d data/SLI/output/printed/rotated -o data/SLI/output

   # Individual steps
   python scripts/processing/classifiers.py -j data/SLI/input -o data/SLI/output

Advanced Options
~~~~~~~~~~~~~~~~

**Detection Parameters:**

.. code-block:: bash

   python scripts/processing/detection.py \
     -j data/MLI/input \
     -o data/MLI/output \
     --confidence 0.5 \
     --batch-size 16 \
     --device auto \
     --no-cache        # optional
   # Cache maintenance
   python scripts/processing/detection.py --clear-cache

**OCR Configuration:**

.. code-block:: bash

   # Tesseract (printed labels after rotation)
   python scripts/processing/tesseract.py \
     -d data/SLI/output/printed/rotated \
     -o data/SLI/output \
     -t 1            # 1=Otsu, 2=Adaptive-Mean, 3=Adaptive-Gaussian
   
   # Google Vision (printed labels after rotation)
   python scripts/processing/vision.py \
     -c credentials.json \
     -d data/SLI/output/printed/rotated \
     -o data/SLI/output

Manual Pipeline Scripts
------------------------

Direct Script Execution
~~~~~~~~~~~~~~~~~~~~~~~~

For advanced users or batch processing, run pipeline scripts directly:

.. code-block:: bash

   # Multi-label pipeline (conda-based)
   ./tools/pipelines/run_mli_pipeline_conda.sh

   # Single-label pipeline (conda-based)
   ./tools/pipelines/run_sli_pipeline_conda.sh

   # Set custom input/output paths
   INPUT_DIR=/path/to/input OUTPUT_DIR=/path/to/output ./tools/pipelines/run_mli_pipeline_conda.sh

Benefits of Direct Scripts:
- Full control over environment
- Custom path configuration
- Batch processing integration
- Debugging and development

Understanding Results
---------------------

Output Structure
~~~~~~~~~~~~~~~~

**Multi-Label Results:**

.. code-block:: text

   data/MLI/output/
   ├── input_predictions.csv          # Detection coordinates and confidence
   ├── input_cropped/                 # Individual label images
   ├── detection_stats.json           # Processing statistics
   └── consolidated_results.json      # Complete detection report

**Single-Label Results:**

.. code-block:: text

   data/SLI/output/
   ├── classification/
   │   ├── empty/                     # Empty labels
   │   ├── handwritten/               # Handwritten labels
   │   ├── printed/                   # Printed labels
   │   └── identifier/                # QR codes, barcodes
   ├── ocr_results/
   │   ├── tesseract/                 # Tesseract OCR output
   │   └── google_vision/             # Google Vision API output
   ├── processed/
   │   ├── corrected_transcripts.json # Cleaned and corrected text
   │   ├── plausible_transcripts.json # High-confidence results
   │   └── metadata.json              # Processing metadata
   └── consolidated_results.json      # Final structured output

Key Output Files
~~~~~~~~~~~~~~~~

**consolidated_results.json**
   Complete processing results including:
   - Original image metadata
   - Detection/classification results
   - OCR transcriptions
   - Confidence scores
   - Processing timestamps

**corrected_transcripts.json**
   Post-processed text with:
   - Spelling corrections
   - Format standardization
   - Entity extraction
   - Confidence ratings

**plausible_transcripts.json**
   High-quality extractions suitable for:
   - Automated database entry
   - Research analysis
   - Publication-ready data

Quality Assessment
~~~~~~~~~~~~~~~~~~

**Confidence Scores:**
- Detection confidence: Probability of correct label detection
- Classification confidence: Accuracy of label type identification
- OCR confidence: Text extraction reliability

**Quality Indicators:**
- Image resolution and clarity
- Text contrast and legibility
- Processing success rates
- Manual review recommendations

Processing Workflows
--------------------

Complete Museum Digitization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Image Capture**

   .. code-block:: bash

      # Photograph specimens with multiple labels
      # Save as high-resolution JPEG files

2. **Multi-Label Detection**

   .. code-block:: bash

      python scripts/processing/detection.py -j photos/ -o detections/

3. **Label Extraction**

   .. code-block:: bash

      # Move cropped labels to SLI pipeline
      cp detections/input_cropped/* data/SLI/input/

4. **Single-Label Processing**

   .. code-block:: bash

      python scripts/processing/analysis.py -j data/SLI/input -o data/SLI/output

5. **Quality Control**

   .. code-block:: bash

      python scripts/evaluation/analysis_eval.py -i data/SLI/output/

Research Data Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Direct Processing**

   .. code-block:: bash

      # Process pre-cropped research labels
      python scripts/processing/analysis.py -j research_labels/ -o results/

2. **High-Confidence Filtering**

   .. code-block:: bash

      # Extract reliable data
      jq '.[] | select(.confidence > 0.8)' results/plausible_transcripts.json

3. **Data Export**

   .. code-block:: bash

      # Convert to CSV for analysis
      python scripts/postprocessing/consolidate_results.py -i results/ -f csv

Batch Processing
~~~~~~~~~~~~~~~~

For large datasets:

.. code-block:: bash

   # Process in batches of 50 images
   find data/MLI/input -name "*.jpg" | split -l 50 - batch_

   # Process each batch
   for batch in batch_*; do
       mkdir batch_input batch_output
       while read img; do cp "$img" batch_input/; done < "$batch"
       python scripts/processing/detection.py -j batch_input -o batch_output
       # Consolidate results
   done

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Low Detection Accuracy**
- Check image quality and resolution
- Adjust confidence thresholds
- Verify lighting and contrast
- Consider manual cropping for difficult cases

**OCR Errors**
- Try different OCR methods (Tesseract vs Google Vision)
- Adjust language settings
- Check for proper rotation correction
- Review image preprocessing steps

**Memory Issues**
- Reduce batch sizes
- Process images sequentially
- Close other applications
- Consider using Docker for memory management

**Performance Problems**
- Use GPU acceleration when available
- Optimize image sizes
- Process in smaller batches
- Monitor system resources

Getting Help
~~~~~~~~~~~~

When encountering issues:

1. Check log files for error messages
2. Verify input data format and quality
3. Test with sample images first
4. Consult the troubleshooting documentation
5. Report issues with detailed error information

Best Practices
--------------

Image Preparation
~~~~~~~~~~~~~~~~~

- Standardize lighting conditions
- Maintain consistent resolution
- Remove dust and debris from labels
- Ensure labels are flat and unfolded

Processing Strategy
~~~~~~~~~~~~~~~~~~~

- Start with small test batches
- Validate results before large-scale processing
- Keep original images as backups
- Document processing parameters used

Quality Control
~~~~~~~~~~~~~~~

- Review classification results manually
- Validate high-confidence OCR outputs
- Check for systematic errors
- Maintain processing logs

Data Management
~~~~~~~~~~~~~~~

- Organize results by processing date
- Archive original images separately
- Document metadata and provenance
- Plan for long-term data storage

Advanced Features
-----------------

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

Create custom processing configurations:

.. code-block:: python

   # config/custom_settings.py
   DETECTION_CONFIDENCE = 0.85
   OCR_METHOD = 'google'
   LANGUAGE = 'eng+fra'  # Multi-language support
   OUTPUT_FORMAT = 'json'

Programmatic Access
~~~~~~~~~~~~~~~~~~~

Use the system programmatically:

.. code-block:: python

   from label_processing import LabelProcessor

   processor = LabelProcessor()
   results = processor.process_directory('data/SLI/input')
   processor.save_results(results, 'output.json')

Integration
~~~~~~~~~~~

Integrate with existing systems:

.. code-block:: python

   # Database integration example
   import json
   from your_database import Database

   with open('consolidated_results.json') as f:
       data = json.load(f)

   db = Database()
   for record in data:
       db.insert_specimen_data(record)