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

Gemini Pipeline (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gemini pipeline uses the Google Gemini API for all vision tasks. It is the recommended pipeline for most users and handles both printed and handwritten labels:

1. **Detection + Classification**: Gemini detects all labels in a specimen image, classifies them (printed, handwritten, mixed, identifier, empty), and determines rotation angle
2. **OCR / HTR**: Gemini reads text from each label (works for printed AND handwritten)
3. **Post-processing**: Text cleaning and consolidation
4. **Entity Recognition**: Gemini extracts structured entities (scientific names, collectors, dates, localities)
5. **GBIF + OSM Enrichment**: Validates names against GBIF Backbone Taxonomy and geocodes localities with OpenStreetMap
6. **Darwin Core Export**: Outputs standardised Darwin Core records (JSON and CSV)
7. **Crop & Cleanup**: Optional label cropping and intermediate file removal

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

* ``gemini.txt`` - Dependencies for the Gemini pipeline (lightweight: google-genai, opencv, pandas, requests, numpy, nltk)
* ``classifier.txt`` - Dependencies for classification models (traditional)
* ``empty_labels.txt`` - Dependencies for empty label detection (traditional)
* ``postprocess.txt`` - Dependencies for text post-processing
* ``rotation.txt`` - Dependencies for rotation correction (traditional)
* ``segmentation.txt`` - Dependencies for label segmentation (traditional)
* ``tesseract.txt`` - Dependencies for Tesseract OCR (traditional)

Docker Usage
------------

To run the pipelines:

.. code-block:: bash

   # Gemini pipeline (recommended — lightweight, API-based)
   cd pipelines
   GEMINI_API_KEY=<your-key> docker-compose --profile gemini up

   # Multi-label processing (MLI, traditional)
   cd pipelines
   docker-compose --profile mli up

   # Single-label processing (SLI, traditional)
   cd pipelines
   docker-compose --profile sli up
   
   # Run individual services (traditional)
   cd pipelines
   docker-compose up segmentation  # Detection only
   docker-compose up rotation      # Rotation correction only
   docker-compose up tesseract     # OCR only
   docker-compose up classification_nuri  # ID/Description classification
   docker-compose up classification_hp    # Handwritten/Printed classification
