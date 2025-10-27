pipelines Package
==================

The ``pipelines`` package provides Docker-based processing pipelines and workflow management.

.. currentmodule:: pipelines

Package Contents
----------------

This package contains a unified Docker configuration and requirements for different processing pipelines:

* ``Dockerfile`` - Consolidated multi-stage Dockerfile for all pipeline components
* ``docker-compose.yml`` - Unified Docker Compose with profiles for MLI, SLI, and standalone services
* ``requirements/`` - Directory containing specific requirements for different pipeline components

Pipeline Configurations
------------------------

Multi-Label Pipeline
~~~~~~~~~~~~~~~~~~~~

The multi-label pipeline processes full specimen images with multiple labels:

1. **Label Detection**: Uses Faster R-CNN to detect individual labels
2. **Label Cropping**: Extracts detected labels as separate images  
3. **Classification**: Determines label types (empty, handwritten, printed, identifier)
4. **Processing**: Routes labels for appropriate processing

Single-Label Pipeline  
~~~~~~~~~~~~~~~~~~~~~

The single-label pipeline processes pre-cropped individual label images:

1. **Classification**: Determines label types
2. **Rotation Correction**: Corrects label orientation
3. **OCR Processing**: Extracts text using Tesseract or Google Vision API
4. **Post-processing**: Cleans and structures extracted text

Requirements Structure
~~~~~~~~~~~~~~~~~~~~~~

The ``requirements/`` directory contains specialized dependency files:

* ``classifier.txt`` - Dependencies for classification models
* ``empty_labels.txt`` - Dependencies for empty label detection
* ``postprocess.txt`` - Dependencies for text post-processing
* ``rotation.txt`` - Dependencies for rotation correction
* ``segmentation.txt`` - Dependencies for label segmentation
* ``tesseract.txt`` - Dependencies for Tesseract OCR

Docker Usage
------------

To run the pipelines:

.. code-block:: bash

   # Multi-label processing (MLI)
   cd pipelines
   docker-compose --profile mli up

   # Single-label processing (SLI)
   cd pipelines
   docker-compose --profile sli up
   
   # Run individual services
   cd pipelines
   docker-compose up segmentation  # Detection only
   docker-compose up rotation      # Rotation correction only
   docker-compose up tesseract     # OCR only
   docker-compose up classification_nuri  # ID/Description classification
   docker-compose up classification_hp    # Handwritten/Printed classification
