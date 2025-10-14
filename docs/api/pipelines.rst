pipelines Package
==================

The ``pipelines`` package provides Docker-based processing pipelines and workflow management.

.. currentmodule:: pipelines

Package Contents
----------------

This package contains Docker Compose configurations and requirements for different processing pipelines:

* ``multi-label-docker-compose.yaml`` - Multi-label image processing pipeline
* ``single-label-docker-compose.yaml`` - Single-label image processing pipeline
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

   # Multi-label processing
   docker-compose -f pipelines/multi-label-docker-compose.yaml up

   # Single-label processing  
   docker-compose -f pipelines/single-label-docker-compose.yaml up